# probe_sdf.py
import torch, numpy as np
from graspqp.hands import get_hand_model

pts = torch.tensor([
    [0.19037975370883942, -0.1640506237745285, 0.1487341821193695],
    [0.2346835434436798,  -0.20202530920505524, 0.15506328642368317],
], dtype=torch.float32)

hand = get_hand_model("linkerhand_l20", "cpu")
pose = torch.cat([torch.zeros(3), torch.tensor([0,0,0,1,0,0], dtype=torch.float), hand.default_state]).unsqueeze(0)
hand.set_parameters(pose, contact_point_indices="all")

d, li = hand.cal_distance(pts.unsqueeze(0), return_link_index=True)
d = d[0].detach().cpu().numpy()
li = li[0].detach().cpu().numpy()
print("link_names:", hand.link_names)
for i in range(len(pts)):
    lname = hand.link_names[li[i]] if li[i] < len(hand.link_names) else f"link_{li[i]}"
    print(f"  point[{i}]: sdf={d[i]*1000:.3f} mm  -> {lname}")