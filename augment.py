import os
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from os.path import isfile, join, isdir

root_dir = "./data/"
csv_file = "./data/dogs.csv"

new_root_dir = "./new_data/"
new_csv_file = "./new_data/dogs.csv"

transform_resize = transforms.Compose([transforms.Resize((224,224)),])

transform_augment = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomErasing(p=0.3, scale=(0.02,0.1),ratio=(1,1)),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=60),
    ], p=0.3),
    transforms.RandomApply([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    ], p=0.3),
    transforms.ToPILImage(),
])


with open(new_csv_file, 'w', newline='') as csvfile:
    csvfile.write("filepaths,labels,data set\n")
    for dir in next(os.walk(root_dir))[1]:
        for racedir in next(os.walk(root_dir + dir))[1]:
            Path(new_root_dir+"/".join([dir,racedir])).mkdir(parents=True, exist_ok=True)
            for file in next(os.walk(root_dir + dir + "/" + racedir))[2]:
                filename = "/".join([dir,racedir,file])
                img = Image.open(root_dir + filename)

                img_resize = transform_resize(img)
                line = ",".join([filename,racedir,dir])+"\n"
                csvfile.write(line)
                img_resize.save(join(new_root_dir, filename))



                abc = ["a","b","c","d","e","f","g","h","i","j"]

                for letter in abc:
                    img_augmented = transform_augment(img_resize)

                    x = file.split(".")
                    x.insert(1, letter)
                    augmented_filename = ".".join(x)
                    augmented_filename = "/".join([dir,racedir,augmented_filename])
                    line = ",".join([augmented_filename,racedir,dir]) + "\n"
                    csvfile.write(line)
                    img_augmented.save(join(new_root_dir, augmented_filename))



                print(f'Augmented Image: {filename}')


