# Question 1:
import random
import matplotlib.pyplot as plt
from tqdm import trange

def buffon_circle_probability1(line_spacing, num_trials):
    crossings = 0  # Number of times the circle crosses a line
    
    for _ in trange(num_trials):
        # Randomly generate the radius & y of the circle
        R = random.uniform(0, line_spacing / 4)     # R uniformly distributed over the interval [0, d/4]
        y = random.uniform(0, line_spacing * 4)     # y uniformly distributed over [0, 4d]

        # Calculate the range of the integer k
        low_index = int((y - R) // line_spacing)
        high_index = int((y + R) // line_spacing + 1)

        # Check if the circle cross only 1 line
        curr_cross_count = 0
        for i in range(low_index, high_index + 1):
            if i * line_spacing <= y + R and i * line_spacing >= y - R:
                curr_cross_count += 1
        
        if curr_cross_count == 1:
            crossings += 1
                    
    # Calculate the probability
    probability = crossings / num_trials
    return probability

line_spacing = 5.0  # Distance between the parallel lines
num_trials_list = [10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000, 1000000, 10000000]
probability_list = []

for num_trials in num_trials_list:
    probability_list.append(buffon_circle_probability1(line_spacing, num_trials))

plt.semilogx(num_trials_list, probability_list, marker = ".", label = '$a_n = estimated \quad P(Z=1)$')
plt.xlabel('$n$')
plt.ylabel('$maginitude$')
plt.legend()
plt.title('Buffon\'s Circle Question1')
plt.show()

print("The estimated P(Z=1): " + str(probability_list[-1]))

# Question 2:
def buffon_circle_probability2(line_spacing, num_trials):
    cross_1 = 0     # Number of times the circle crossed only 1 line
    crossings = 0    # Number of times the circle crosses a line
    
    for _ in trange(num_trials):
        # Randomly generate the y and radius of the circle
        y = random.uniform(0, line_spacing * 4)    # y uniformly distributed over [0, 4d]
        R = random.expovariate(2 / line_spacing)    # R exponentially distributed with mean d / 2

        # Calculate the range of the integer k
        low_index = int((y - R) // line_spacing)    
        high_index = int((y + R) // line_spacing + 1)

        # Check if the circle cross only 1 line and the total crossings
        curr_cross = 0
        for i in range(low_index, high_index + 1):
            if i * line_spacing <= y + R and i * line_spacing >= y - R:
                curr_cross += 1
                crossings += 1
        if curr_cross == 1:
            cross_1 += 1
                    
    # Calculate the expectation and probability
    exp = crossings / num_trials
    prob = cross_1 / num_trials
    return exp, prob

line_spacing = 5.0  # Distance between the parallel lines
num_trials_list = [10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000, 1000000, 10000000]
exp_list = []
prob_list = []

for num_trials in num_trials_list:
    exp, prob = buffon_circle_probability2(line_spacing, num_trials)
    exp_list.append(exp)
    prob_list.append(prob)

plt.semilogx(num_trials_list, prob_list, marker = "*", label = '$b_n = estimated \quad P(Z=1)$')
plt.semilogx(num_trials_list, exp_list, marker = ".", label = '$c_n = estimated \quad E[Z]$')
plt.xlabel('n')
plt.ylabel('magnitude')
plt.title('Buffon\'s Circle Question2')
plt.legend()
plt.show()

print('The estimated E[Z]: ' + str(exp_list[-1]))
print('The estimated P(Z=1): ' + str(prob_list[-1]))