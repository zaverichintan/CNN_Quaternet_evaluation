import imageio

path = './visualization/posing/'
with imageio.get_writer('visualization/posing.gif', mode='I') as writer:
    for filename in range(500):
        image = imageio.imread(path+str(filename)+'.png')
        writer.append_data(image)