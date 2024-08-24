import torch
import datasets
import networks

def get_monovit_pretrained(load_type):
    depth = networks.DeepNet('mpvitnet')
    if load_type == 1024:
        depth_dict = torch.load(r"F:\rd\mobilevit_distilla_pose_nofreeze_light_decoder\models\1024_320\depth.pth")
        new_dict = depth_dict
        new_dict = {}
        for k, v in depth_dict.items():
            name = k[7:]
            new_dict[name] = v

        depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})

    if load_type == 640:
        # 加载编码器的权重
        mpvit_encoder_dict = torch.load(r"F:\rd\mobilevit_distilla_pose_nofreeze_light_decoder\models\monovit640_stereo\encoder.pth")
        model_dict = depth.encoder.state_dict()
        depth.encoder.load_state_dict({k: v for k, v in mpvit_encoder_dict.items() if k in model_dict})
        # 加载解码器的权重
        mpvit_decoder_path = r"F:\rd\mobilevit_distilla_pose_nofreeze_light_decoder\models\monovit640_stereo\depth.pth"
        depth.decoder.load_state_dict(torch.load(mpvit_decoder_path))

    return depth
