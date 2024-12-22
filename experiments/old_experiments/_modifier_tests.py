from fixkit.localization.modifier import *


if __name__ == "__main__":
    identifiers = [WeightedIdentifier(i, x) for i, x in zip(range(1, 6), [0.0, 0.0, 0.0, 0.0, 0.0])]
    print(identifiers)
    modifier = TopEqualRankModifier()
    locations = modifier.locations(identifiers)
    print(locations)
    for location in locations:
        print(modifier.mutation_chance(location))