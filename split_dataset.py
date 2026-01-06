import os
from sklearn.model_selection import train_test_split

images_dir = './images'
all_names = sorted([os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

train_names, test_names = train_test_split(all_names, test_size=0.3, random_state=42)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42)

def save_list(filename, names):
    with open(filename, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

save_list("train_files.txt", train_names)
save_list("val_files.txt", val_names)
save_list("test_files.txt", test_names)

print(f"Created manifest files. Test set size: {len(test_names)}")