import os
import cv2
import torch
import time
import datetime
import pickle as pkl

from model.utils.data_viewer import dataViewer
from integrase import INTEGRASE, classes
from model.utils.net_utils import relscores_to_visscores, inner_loop_planning
from robot_demo_ingress_vmrn import save_visualization, split_long_string

def ground_with_outerloop_updater(s_ing_client):
    all_results = []
    im_id = raw_input("image ID: ")
    expr = raw_input("Please tell me what you want: ")
    while(True):
        related_classes = [cls for cls in classes if cls in expr or expr in cls]
        img = cv2.imread("images/" + im_id + ".png")
        data_viewer = dataViewer(classes)

        bboxes, scores, rel_mat, rel_score_mat, leaf_desc_prob, ground_score, ground_result, ind_match = \
            s_ing_client.single_step_perception_new(img, expr, cls_filter=related_classes)
        num_box = bboxes.shape[0]

        # dummy action for initialization
        a = 3 * num_box + 1
        # outer-loop planning: in each step, grasp the leaf-descendant node.
        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)
        belief = {}
        belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_prob)
        belief["ground_prob"] = torch.from_numpy(ground_result)

        # inner-loop planning, with a sequence of questions and a last grasping.
        while (True):
            a = inner_loop_planning(belief)
            current_date = datetime.datetime.now()
            image_id = "{}-{}-{}-{}".format(current_date.year, current_date.month, current_date.day,
                                            time.strftime("%H:%M:%S"))
            all_results.append(
                save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, data_viewer,
                                   im_id=image_id))
            if a < 2 * num_box:
                s_ing_client.object_pool[ind_match[a % num_box]]["removed"] = True
                break
            else:
                data = {"img": img,
                        "bbox": bboxes[:, :4].reshape(-1).tolist(),
                        "cls": bboxes[:, 4].reshape(-1).tolist()}
                ans = raw_input("Your answer: ")
                all_results[-1]["answer"] = split_long_string("User's Answer: " + ans.upper())

                if a < 3 * num_box:
                    # we use binary variables to encode the answer of q1 questions.
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        for i, v in enumerate(s_ing_client.object_pool):
                            s_ing_client.object_pool[i]["confirmed"] = True
                            if i == ind_match[a - 2 * num_box]:
                                s_ing_client.object_pool[i]["ground_belief"] = 1.
                            else:
                                s_ing_client.object_pool[i]["ground_belief"] = 0.
                    else:
                        obj_ind = ind_match[a - 2 * num_box]
                        s_ing_client.object_pool[obj_ind]["confirmed"] = True
                        s_ing_client.object_pool[obj_ind]["ground_belief"] = 0.
                else:
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        s_ing_client.target_in_pool = True
                        for i, v in enumerate(s_ing_client.object_pool):
                            if i not in ind_match.values():
                                s_ing_client.object_pool[i]["confirmed"] = True
                                s_ing_client.object_pool[i]["ground_belief"] = 0.
                    else:
                        # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                        ans = ans[9:]
                        for i in ind_match.values():
                            s_ing_client.object_pool[i]["confirmed"] = True
                            s_ing_client.object_pool[i]["ground_belief"] = 0.
                        s_ing_client.clue = ans

                belief = s_ing_client.update_belief(belief, a, ans, data)
        break

    return ground_score, rel_score_mat

if __name__ == '__main__':
    s_ing_client = INTEGRASE()
    # new_main(s_ing_client)

    if os.path.exists("density_esti_train_data.pkl"):
        with open("density_esti_train_data.pkl", "rb") as f:
            saved_data = pkl.load(f)
        gr_data = saved_data["ground"]
        rel_data = saved_data["relation"]
    else:
        gr_data = []
        rel_data = []

    while(True):
        ground_score, rel_score = ground_with_outerloop_updater(s_ing_client)
        gr_id_str = raw_input("Tell me the ground truth index: ")
        gr_ids = gr_id_str.split(" ")
        gr_data.append({"scores": ground_score[:-1], "gt": gr_ids})
        rel_data.append(rel_score)
        with open("density_esti_train_data.pkl", "wb") as f:
            pkl.dump({"ground": gr_data, "relation": rel_data}, f)
    pass

