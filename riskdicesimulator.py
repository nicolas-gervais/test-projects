import numpy as np

print()
attacking_units = int(input('Number of units on the ATTACKING territory: '))
attacking_dices = int(input('Number of dices used in the ATTACK: '))

if attacking_dices < 1 or attacking_dices > 3:
    print('INPUT ERROR: you can only attack with 1, 2, or 3 dices at a time. Start over.')

defending_units = int(input('Number of units on the DEFENDING territory: '))
defending_dices = int(input('Number of dices used in the DEFENSE: '))

if defending_dices < 1 or defending_dices > 2:
    print('INPUT ERROR: you can only defend with 1 or 2 dices at a time. Start over.')

print(); print("Computing...")


def dice_roll(number):
    rolls = []
    for n in range(number):
        rolls.append(np.random.randint(1, 7))
    return rolls


def dice_engine(attacking_units, defending_units):

    att_units = attacking_units
    def_units = defending_units

    def number_attack_dices():
        if att_units >= 4 and attacking_dices == 3:
            return 3
        elif att_units >= 3 and attacking_dices == 2:
            return 2
        else:
            return 1

    def number_defense_dices():
        if def_units >= 2 and defending_dices == 2:
            return 2
        else:
            return 1

    while att_units > 1 and def_units > 0:
        attacker_roll = np.sort(dice_roll(number_attack_dices()))
        defender_roll = np.sort(dice_roll(number_defense_dices()))
        if attacker_roll[-1] > defender_roll[-1]:
            def_units -= 1
        else:
            att_units -= 1

        if defender_roll.size < 2 or attacker_roll.size < 2:
            continue
        else:
            if attacker_roll[-2] > defender_roll[-2]:
                def_units -= 1
            else:
                att_units -= 1

    if att_units <= 1:
        return 0
    elif def_units == 0:
        return 1


success = 0

for i in range(10000):
    success += dice_engine(attacking_units, defending_units)
print(); print("The chance of a successful attack is: " + str(np.round(success/10000*100, 2)) + "%.")


