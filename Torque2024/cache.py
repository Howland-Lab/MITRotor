import functools
from pathlib import Path
import polars as pl
import pickle


def cache_pickle(cache_file: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)
            
            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                with open(cache_filepath, "rb") as file:
                    return pickle.load(file)
            else:
                # Generate and save the data
                data = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                with open(cache_filepath, "wb") as file:
                    pickle.dump(data, file)
                return data

        return wrapper

    return decorator


def cache_polars(cache_file: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)
            
            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                return pl.read_csv(cache_filepath)
            else:
                # Generate and save the data
                df = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                df.write_csv(cache_filepath)
                return df

        return wrapper

    return decorator
