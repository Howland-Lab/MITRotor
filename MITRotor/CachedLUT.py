from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import polars as pl
from scipy.interpolate import RectBivariateSpline

class CachedLUT(ABC):
    """
    Abstract base class for a Cached Look-Up Table (LUT) with interpolation capabilities.

    Attributes:
    ----------
    key1 : str
        First key used for table indexing.
    key2 : str
        Second key used for table indexing.
    to_interp : list[str]
        List of column names in the table to perform interpolation on.
    cache_fn : Path
        Path to the CSV file used for caching the table data.
    regenerate : bool, optional
        Flag indicating whether to regenerate the table even if cache file exists (default is False).
    s : float, optional
        Smoothing factor used in interpolation (default is 0.001).

    Methods:
    --------
    load() -> pl.DataFrame:
        Load the cached table data from the CSV file.

    save() -> pl.DataFrame:
        Save the current table data to the CSV file.

    _make_interpolator_on(on: str) -> RectBivariateSpline:
        Create a 2D interpolation function for a specific column ('on') in the table.

    make_interpolators() -> dict[str, RectBivariateSpline]:
        Create interpolation functions for all columns specified in 'to_interp'.

    generate_table() -> pl.DataFrame:
        Abstract method to generate the look-up table. To be implemented in subclasses.
    """

    def __init__(
        self,
        key1: str,
        key2: str,
        to_interp: list[str],
        cache_fn: Path,
        regenerate: bool = False,
        s=0.001,
    ):
        """
        Initialize CachedLUT with specified parameters.

        Parameters:
        -----------
        key1 : str
            First key used for table indexing.
        key2 : str
            Second key used for table indexing.
        to_interp : list[str]
            List of column names in the table to perform interpolation on.
        cache_fn : Path
            Path to the CSV file used for caching the table data.
        regenerate : bool, optional
            Flag indicating whether to regenerate the table even if cache file exists (default is False).
        s : float, optional
            Smoothing factor used in interpolation (default is 0.001).
        """
        self.key1 = key1
        self.key2 = key2
        self.to_interp = to_interp
        self.cache_fn = cache_fn
        self.s = s

        if cache_fn.exists() and not regenerate:
            self.df = self.load()
        else:
            self.df = self.generate_table()
            self.save()

        self.interpolators = self.make_interpolators()

    @abstractmethod
    def generate_table(self) -> pl.DataFrame:
        """
        Abstract method to generate the look-up table.

        This method should be implemented in subclasses to define how the table
        is generated based on specific requirements.

        Returns:
        --------
        pl.DataFrame:
            The generated look-up table.
        """
        pass

    def load(self) -> pl.DataFrame:
        """
        Load the cached table data from the CSV file.

        Returns:
        --------
        pl.DataFrame:
            The loaded DataFrame from the CSV.
        """
        return pl.read_csv(self.cache_fn)

    def save(self) -> None:
        """
        Save the current table data to the CSV file.

        Returns:
        --------
        None
        """
        self.df.write_csv(self.cache_fn)

    def _make_interpolator_on(self, on: str) -> RectBivariateSpline:
        """
        Create a 2D interpolation function for a specific column ('on') in the table.

        Parameters:
        -----------
        on : str
            Column name for which to create the interpolation function.

        Returns:
        --------
        RectBivariateSpline:
            Interpolation function for the specified column.
        """
        df_piv = (
            self.df.sort(self.key1, self.key2)
            .pivot(index=self.key2, columns=self.key1, values=on, maintain_order=True)
            .sort(self.key2)
        )

        x = np.array(df_piv.columns[1:], dtype=float).copy()
        y = df_piv[self.key2].to_numpy().copy()
        z = df_piv.to_numpy()[:, 1:].copy()

        interp = RectBivariateSpline(y, x, z, s=self.s)
        return interp

    def make_interpolators(self) -> dict[str, RectBivariateSpline]:
        """
        Create interpolation functions for all columns specified in 'to_interp'.

        Returns:
        --------
        dict[str, RectBivariateSpline]:
            Dictionary where keys are column names and values are interpolation functions.
        """
        interpolators = {}
        for key in self.to_interp:
            interpolators[key] = self._make_interpolator_on(key)
        return interpolators
