from dataset_utils.corruptions import *
import random

class CustomAugmenter():
    """
    Applies augmentation techniques chosen for the age estimation problem, according to the defined policy
    (that is implicit in the code, it is not managed explicitly). 
    In particular, with the policy implemented two kind of corruptions are identified, frequent and non-frequent ones, and a probability
    is associated to each kind of corruption, so that, given that a corruption must occur, these probabilities
    are used to choose which kind of corruption must be applied. To choose wheter an image must be corrupted,
    another probability is used.
    """
    def __init__(self, corruptions_prob, frequent_corruptions_prob):
        """
        Parameters
        ----------
        corruptions_prob : float 
            The probability of an image being corrupted. Must be in [0,1].

        frequent_corruptions_prob : float 
            The probability of a frequent corruption to happen. Must be in [0,1].
        """
        self.frequent_corruptions = [gaussian_noise, random_crop, brightness, contrast, horizontal_flip, rotation]
        self.other_corruptions = [gaussian_blur, skew, spatter]
        self.corruptions_prob = corruptions_prob
        self.frequent_corruptions_prob = frequent_corruptions_prob

    # data is an array (batch)
    def apply_augmentation(self, data):
        """
        Applies one of the augmentations encapsulated by the class, with the policy encapsulated by the class.
        """
        # can't use foreach-like syntax because it create a reference that is local to the for loop scope, while I need to modify
        # the actual object in the list
        for i in range(0, len(data)):
            # a corruption happens with corruption_prob probability
            if random.uniform(0,1) > self.corruptions_prob:
                return
            # choose if the corruption must be a 'frequent' one or an 'other' one
            if random.uniform(0,1) < self.frequent_corruptions_prob:
                corruptions_list = self.frequent_corruptions
            else:
                corruptions_list = self.other_corruptions
            # choose a random corruption in the list
            corruption_index = random.randint(0, len(corruptions_list)-1) # upper and lower limits are inclusive
            # Set severity for corruptions that require her
            severity = None
            if corruptions_list[corruption_index] == gaussian_noise:
                severity = random.uniform(0.08, 0.15)

            elif corruptions_list[corruption_index] == random_crop:
                severity = random.uniform(0.1, 0.4)

            elif corruptions_list[corruption_index] == brightness:
                severity = random.uniform(0.1, 0.5)
                if random.uniform(0, 1) < 0.5:
                    severity = 0 - severity

            elif corruptions_list[corruption_index] == contrast:
                if random.uniform(0,1) < 0.5:
                    severity = random.uniform(1.5, 5.0)
                else:
                    severity = random.uniform(0.1, 0.4)

            elif corruptions_list[corruption_index] == gaussian_blur:
                severity = random.uniform(1.0, 2.5)

            elif corruptions_list[corruption_index] == spatter:
                severity = random.randint(1,5)
            
            elif corruptions_list[corruption_index] == rotation:
                severity = random.uniform(1,10)
                if random.uniform(0, 1) < 0.5:
                    severity = 0 - severity
            # call chosen corruption on our sample
            if severity is not None:
                data[i] = corruptions_list[corruption_index](data[i], severity)
            else:
                data[i] = corruptions_list[corruption_index](data[i])
        
