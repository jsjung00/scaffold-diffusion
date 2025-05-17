
import nbtlib
import json 

def nbt_test():
    '''
    See what the nbt file looks like
    '''
    nbt_file = nbtlib.load('/home/justinsoljung/voxeldiffusion/output_files/house_2.nbt')

    breakpoint()

    #with open('golf.json', "r") as f:
    #    data = json.load(f)

    #top_keys = data.keys()
    #breakpoint() 


if __name__ == "__main__":
    nbt_test()