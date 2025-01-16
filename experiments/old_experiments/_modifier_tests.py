from fixkit.localization.modifier import *


if __name__ == "__main__":
    identifiers = [WeightedIdentifier(i, x) for i, x in zip(range(1, 9), [1.0, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0])]
    print(identifiers)
    modifier = WeightedTopRankModifier()
    locations = modifier.locations(identifiers)
    print(locations)
    for location in locations:
        print(modifier.mutation_chance(location))