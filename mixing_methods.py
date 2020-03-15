import numpy as np


""" 
Different mixing methods for the ablation analysis
Legend for additional changes to the image mixing equations
 (a) we subtract per-image mean;
 (b) we divide the mixed image by a factor that considers that waveform energy is proportional to the square of the amplitude
 (c) we take the difference of image energies into consideration
"""

# BC (no addition)
def ablation_mix(image1, image2):

    # If you want to subtract mean for normal BC mixing, 
    # then this methods calls a method that performs this

    # Mix images
    r = np.array(random.random())
    image = (image1 * r + image * (1 - r)).astype(np.float32)

    return image

# Implements (a)
def ablation_mix_a(image1, image2, r = None):

    
    ### (a)
    # Get mean of images
    mean1 = np.mean(image1, keepdims=True)
    mean2 = np.mean(image2, keepdims=True)

    # Subtract mean
    image1 = image1 - mean1
    image2 = image2 - mean2

    ###

    # Mix images
    if r == None:
        r = np.array(random.random())
    image = (image1 * r + image * (1 - r)).astype(np.float32)

    return image

# Implements (a), (b)
def ablation_mix_ab(image1, image2): 

    r = np.array(random.random())

    # Get image with (a) criteria
    image = ablation_mix_a(image1, image2, r = r)

    ### (b)
    image = image / np.sqrt( r**2 + (1 - r)**2 )
    ###

    return image

# Implements (b), (c)
# Do not confuse with BC+ mixing, this is for ablation analysis
def ablation_mix_bc(image1, image2, r = None): 

    if r == None:
        r = np.array(random.random())

    ### (b) (c)
    std1 = np.std(image1)
    std2 = np.std(image2)
    p = 1 / ( 1 + std1/std2 * (1 - r)/r )
    
    image = (image1 * p + image * (1 - p)).astype(np.float32)
    image = image / np.sqrt( p**2 + (1 - p)**2 )
    ###

    return image

# Implements (a), (b),(c)
# Same as BC+ mixing
def ablation_mix_abc(image1, image2): 

   
    r = np.array(random.random())

    ### (a)
    mean1 = np.mean(image1, keepdims=True)
    mean2 = np.mean(image2, keepdims=True)

    image1 = image1 - mean1
    image2 = image2 - mean2
    ###


    ### Adds (b) and (c) critiera
    image = ablation_mix_bc(image1, image2, r = r)
    ###

    return image