import os, subprocess

if __name__ == "__main__":
    move_into_container = list()
    if input("Do you want to move some of your local files into to container? This will overwrite files from origin/master. (y/n) ").startswith("y"):
        for f in sorted(os.listdir()):
            if input("Move %s into container (y/n)? " % f).startswith("y"):
                move_into_container.append(f)
    if move_into_container:
        subprocess.call(["tar", "-czvf", "move_into_container.tar.gz"] + move_into_container)
    image_name = input("Name of Image? (Default: Auto-PyTorch.simg) ") or "Auto-PyTorch.simg"
    if os.path.exists(image_name) and input("%s exists. Remove (y/n)? " % image_name).startswith("y"):
        os.remove(image_name)
    print("Building Singularity container. You need to be root for that.")
    subprocess.call(["sudo", "singularity", "build", image_name, "scripts/Singularity"])
    if move_into_container:
        os.remove("move_into_container.tar.gz")
