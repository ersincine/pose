import numpy as np
from extract_pose import extract_pose
from estimate_pose import estimate_pose
from calculate_error import calculate_error
from tqdm import tqdm


# dataset = "Barn"
# method = "nn"

for dataset in ["Barn", "Truck", "Meetingroom"]:
    for method in ["snn", "nn"]:

        num_images_dict = {
            "Barn": 410,
            "Truck": 251,
            "Meetingroom": 371
        }
        num_images = num_images_dict[dataset]
        window_size = 10

        q_errs = []
        t_errs = []

        num_pairs = 0

        for img1_no in tqdm(range(window_size + 1, num_images - window_size, window_size)):
            for img2_no in range(max(1, img1_no - window_size), min(num_images, img1_no + window_size)):
                if img1_no == img2_no:
                    continue

                num_pairs += 1
                extract_pose(img1_no, img2_no, dataset, save=True)
                estimate_pose(img1_no, img2_no, dataset, method, save=True, verbose=False)
                q_err, t_err = calculate_error(img1_no, img2_no, dataset, method)

                q_err = np.rad2deg(q_err)
                t_err = np.rad2deg(t_err)

                q_errs.append(q_err)
                t_errs.append(t_err)
                    
        print(num_pairs)
        print(f"q_err: {min(q_errs)=}, {max(q_errs)=}, {sum(q_errs) / len(q_errs)=}")
        print(f"t_err: {min(t_errs)=}, {max(t_errs)=}, {sum(t_errs) / len(t_errs)=}")

        np.savetxt(f"{dataset}_{method}_q_errs.txt", q_errs)
        np.savetxt(f"{dataset}_{method}_t_errs.txt", t_errs)

        """
        # Barn
        q_err: min(q_errs)=0.0025643270094704938, max(q_errs)=3.1372744276217115, sum(q_errs) / len(q_errs)=0.8400769632233732
        t_err: min(t_errs)=0.02859372482259649, max(t_errs)=1.5695622940387, sum(t_errs) / len(t_errs)=0.871715180409327

        0.85 rad = 48.7 deg
        """
