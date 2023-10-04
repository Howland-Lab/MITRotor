def test_example_001():
    from examples.example_001_momentum_aligned import main

    main()


def test_example_002():
    from examples.example_002_blah import main

    main()


def test_example_003():
    from examples.example_003_iterations import main

    main()


def test_example_004():
    from examples.example_004_ThrustInduction import func

    args_to_test = [
        (0.0, 2, "UnifiedMomentum"),
        (0.0, 2, "Heck"),
        (0.0, 2, "Madsen"),
    ]

    for args in args_to_test:
        func(args)
