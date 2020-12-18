"""
Build co-occurence matrix
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import scipy.spatial
import os, sys

def cooccurence_calculate(manifest, names):
    
    # all categories between [0,79]
    with open(manifest, "rt") as f:
        manifest = f.readlines()
        manifest = [elem.strip() for elem in manifest]
        manifest = [elem.replace("images","labels") for elem in manifest]
        manifest = [elem.replace(".jpg",".txt") for elem in manifest]

    num_categories = 80
    cooccurence    = np.zeros(shape=(num_categories,num_categories), dtype=np.float32)

    all_targets = []
    for label in manifest:
        targets = np.loadtxt(label, delimiter=" ")
        targets = np.reshape(targets, (-1,5))
        all_targets.append(targets)

    flat_targets = np.concatenate(all_targets, axis=0)
    all_categories = np.unique(flat_targets[:,0])

    for targets in all_targets:
        categories = np.unique(targets[:,0])
        for category1 in categories:
            for category2 in categories:
                cooccurence[int(category1),int(category2)] += 1.0

    cooccurence[range(num_categories), range(num_categories)] = 0.0 # set diagonal values to 0
    norm_factor = np.sum(cooccurence, axis=1).reshape((num_categories,1))
    cooccurence = cooccurence / norm_factor
    print(cooccurence.sum(axis=1))

    # names
    with open(names ,"rt") as f:
        coco_names = f.readlines()
        coco_names = [elem.strip() for elem in coco_names]

    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(cooccurence, cmap='jet', interpolation='nearest')
    im.set_clim(0.0, 0.2)
    ax.set_title("Co-occurence of categories (MS-COCO 2014)")
    ax.set_xticks(np.arange(80))
    ax.set_yticks(np.arange(80))
    ax.set_xticklabels(coco_names, rotation='vertical')
    ax.set_yticklabels(coco_names)
    # ax.xticks(np.arange(80), coco_names, rotation='vertical')
    # ax.yticks(np.arange(80), coco_names)
    ax.tick_params(axis='x', top=True, bottom=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis='y', left=True, right=True, labelleft=True, labelright=True)
    fig.tight_layout()
    ax.figure.colorbar(im, ax=ax, extend="max")
    
    return cooccurence, fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="calculate co-occurence matrix of categories in 2 datasets and measure divergence between them")
    parser.add_argument("--src", type=str, default="/tmp/coco/trainvalno5k.txt", help="manifest file for dataset 1")
    parser.add_argument("--gt", type=str, default="/tmp/coco/trainvalno5k.txt", help="manifest file for dataset 1")
    parser.add_argument("--names", type=str, default="/home/achawla/akshayws/yolov3/data/coco.names", help="path to coco.names file")
    parser.add_argument("--outdir", type=str, default='./', help="save outputs to this directory")
    args = parser.parse_args()

    cooccurence_src, fig_src = cooccurence_calculate(args.src, args.names)
    cooccurence_gt,  fig_gt  = cooccurence_calculate(args.gt,  args.names)


    per_category_divergence = np.zeros((80,), dtype=np.float32)
    num_categories = 80
    for category in range(num_categories):
        
        # Make sure it is probability distribution
        # assert cooccurence_src[category,:].sum()==1.0, "sum is: {}".format(cooccurence_src[category,:].sum())
        # assert cooccurence_gt[category,:].sum()==1.0, "sum is: {}".format(cooccurence_gt[category,:].sum())
        # divergence
        per_category_divergence[category] = scipy.spatial.distance.jensenshannon(cooccurence_src[category,:], cooccurence_gt[category,:])

    print("per category divergence: ")
    print(per_category_divergence)

    print("total divergence: {}".format(per_category_divergence.sum()))

    # Write to file 
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    fig_src.savefig(os.path.join(args.outdir, "cooccurence_src.png"))
    fig_gt.savefig(os.path.join(args.outdir, "cooccurence_gt.png"))
    np.savetxt(os.path.join(args.outdir, "cooccurence_src.txt"), cooccurence_src, delimiter=",")
    np.savetxt(os.path.join(args.outdir, "cooccurence_gt.txt"), cooccurence_gt, delimiter=",")
    with open(os.path.join(args.outdir, "divergence.txt"), "wt") as f:
        f.write("category divergence\n")
        for category in range(num_categories):
            f.write("{} {}\n".format(int(category), float(per_category_divergence[category])))
        f.write("{} {}\n".format("all", float(per_category_divergence.sum())))




