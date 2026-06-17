import torch
import numpy as np
import os
from torch.autograd import Function
import importlib
import json
import cv2  # Added for processing structural gradients

LUT_path_root = os.path.join('LUT_test', 'LUTs')

scale, stacks, msb_base, cnum, model_name, step, EPS = (4, 7, 6, 16, 'ShiftLUT_sr_s7_int', 189000, 0.35) # K_Max ~ 64

LUT_path = os.path.join(LUT_path_root, model_name)
os.makedirs(LUT_path, exist_ok=True)

module_name = f'models.{model_name}.model'
Model = importlib.import_module(module_name)

TinyLUTRE = Model.TinyLUTRE
default_cnum = cnum
DepthWise = Model.DepthWise
PointConv = Model.PointConv
UpConv = Model.UpConv

model_G = TinyLUTRE(scale=scale, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum)

lm = torch.load(
    os.path.join('models', model_name, 'Model_{:06d}.pth'.format(step)), weights_only=True)
model_G.load_state_dict(lm, strict=True)
model_G = model_G.cuda()

def save_constant(self):
    combined_tensors = []
    for i in range(stacks+1):
        msb_tensor = self.msb[f"scs{i}"].offset.view(2,16)
        combined_tensors.append(msb_tensor)
    
    result_tensor = torch.stack(combined_tensors)
    print(result_tensor.shape)
    result_tensor = result_tensor.to(torch.int)
    np.save(os.path.join(LUT_path, 'offset.npy'), result_tensor.cpu().numpy())

class Round(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.round(x)
        return out
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

def LUTclip(x):
    return x.clamp(-128, 127)
    
def DW2LUT(m, input_tensor): 
    assert isinstance(m, DepthWise)
    assert (m.stack == 1)
    k, b = m.kernels[0], m.biass[0]
    out = Round.apply(LUTclip(k*input_tensor+b))
    out = out.permute(0, 2, 1, 3)
    out = out[..., 0]
    return out.cpu().numpy().astype(np.int8) 

def PW2LUT(m, input_tensor):
    assert isinstance(m, PointConv)
    ori = [LUTclip(m.conv[i](input_tensor))[:,:,0,0].cpu().numpy() for i in range(default_cnum)]
    return np.stack(ori, 1).astype(np.int8) 

def UP2LUT(m, input_tensor):
    assert isinstance(m, UpConv)
    ori = [LUTclip(m.Conv[i](input_tensor))[:,:,0,0].cpu().numpy() for i in range(default_cnum)]
    return np.stack(ori, 1).astype(np.int8) 

def generate_input(base):
    first_ = base.cuda().unsqueeze(1)
    first__ = torch.cat([first_, first_], 1)
    first___ = torch.cat([first__, first__], 1)
    first_8 = torch.cat([first___, first___], 1)
    first_9 = torch.cat([first_8, first_], 1)
    
    input_tensor_dw = first_9.unsqueeze(1).unsqueeze(1).reshape(-1,1,9,1).float()
    input_tensor_pw = first_.unsqueeze(1).unsqueeze(1).reshape(-1,1,1,1).repeat(1,1,5,5).float()
    return input_tensor_dw, input_tensor_pw


# =========================================================================
# STRATEGY 2 STRUCTURAL INTERVENTIONS START HERE
# =========================================================================

def build_gradient_importance_weights(calibration_folder='dataset/DIV2K_train_LR_bicubic/X4', bit_depth=6):
    """
    Profiles calibration images using Sobel operators to construct an array
    mapping texture/edge density directly onto the 64 potential LUT index inputs.
    """
    max_entries = 2 ** bit_depth  # 64 entries
    gradient_sum = np.zeros(max_entries)
    activation_count = np.zeros(max_entries)
    
    if not os.path.exists(calibration_folder):
        print(f"[*] Calibration path '{calibration_folder}' not found. Defaulting to uniform weights.")
        return np.ones(max_entries)
        
    print("[*] Profiling structural gradients across calibration patches...")
    valid_extensions = ('.png', '.jpg', '.jpeg')
    files = [os.path.join(calibration_folder, f) for f in os.listdir(calibration_folder) if f.lower().endswith(valid_extensions)][:20] # Profile first 20 patches
    
    for file_path in files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Calculate spatial gradient intensities 
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Shift pixel space values to line up with the LUT's 64 index slots (-32 to 32 space shifted)
        shifted_img = (img >> (8 - bit_depth)) # 0 to 63
        
        for idx in range(max_entries):
            mask = (shifted_img == idx)
            if np.any(mask):
                gradient_sum[idx] += np.sum(gradient_mag[mask])
                activation_count[idx] += np.sum(mask)
                
    avg_gradients = np.divide(gradient_sum, activation_count, out=np.zeros_like(gradient_sum), where=activation_count != 0)
    
    if np.max(avg_gradients) > 0:
        # Scale weights dynamically around 1.0. High texture entries get scaling penalties up to 2.0
        weights = 0.5 + (avg_gradients / np.max(avg_gradients)) * 1.5
    else:
        weights = np.ones(max_entries)
        
    return weights

# Pre-generate our localized importance weight matrix
STRUCTURAL_WEIGHTS = build_gradient_importance_weights()


def Query(w, x, sample_d):
    if sample_d == 1:
        return w[x]
    a1 = x//sample_d
    a2 = a1+1
    b = x%sample_d
    weight = b/sample_d
    ans = w[a1*sample_d]*(1-weight) + w[a2*sample_d]*weight
    return ans

# Modified calculation step to include structural weighting elements
def cal_weighted(x, step_a, step_b, weights):
    ans = []
    for i in range(64):
        absolute_diff = np.abs(Query(x, i, step_a) - Query(x, i, step_b))
        # Multiply error by the corresponding structural importance coefficient
        weighted_diff = absolute_diff * weights[i]
        ans.append(weighted_diff)
    return np.mean(ans)


def save_LUT(path, LUT, sample_d):
    N, IN, OUT = LUT.shape
    arrays = {}
    meta = {"IN": IN, "OUT": OUT, "LUTs":[]}
    sum = 0
    
    for in_ch in range(IN):  
        for out_ch in range(OUT):
            data = LUT[:, in_ch, out_ch]
            step = sample_d
            
            if N == 65:
                step = 1
                for candidate_step in [2, 4, 8, 16]:
                    # Call our newly re-engineered texture-aware calc function
                    if cal_weighted(data, 1, candidate_step, STRUCTURAL_WEIGHTS) < EPS*(1-1/candidate_step):
                        step = candidate_step

            if step == 1:
                data = data[:64]
            else:
                data = data[::step]
                
            arrays[f'i{in_ch}o{out_ch}'] = data
            meta["LUTs"].append({"in": in_ch, "out": out_ch, "step": step})
            sum += data.size

    np.savez(path + ".npz", **arrays)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return sum

# =========================================================================
# STRATEGY 2 STRUCTURAL INTERVENTIONS END HERE
# =========================================================================


def Branch2LUT(branch, base, branch_name, stacks, sample_d):
    with torch.no_grad():
        input_tensor_dw, input_tensor_pw = generate_input(base)
        tot = 0
        for i in range(stacks+1):
            res = DW2LUT(branch.dsnets[i*2], input_tensor_dw)
            print(f"For {i} in {branch_name}, Resulting DWLUT size: ", res.shape)
            tot += save_LUT(LUT_path+f"/DW{i}_{branch_name}", res, sample_d[i*2])

            if branch_name == 'MSB':
                res = PW2LUT(branch.dsnets[i*2+1], input_tensor_pw)
                print(f"For {i} in {branch_name}, Resulting PWLUT size: ", res.shape)
                tot += save_LUT(LUT_path+f"/PW{i}_{branch_name}", res, sample_d[i*2+1])

        if branch_name == 'MSB':
            res = UP2LUT(branch.up, input_tensor_pw)
            print(f"For {i} in {branch_name}, Resulting PWLUT size: ", res.shape)
            tot += save_LUT(LUT_path+f"/UP_{branch_name}", res, sample_d[-1])

        print(f"For {branch_name}, storage = {tot/1024}KB.")

save_constant(model_G.module)

msb_steps = [1, 1] * (stacks+1) + [1]
assert (stacks+1)*2+1 == len(msb_steps)

Branch2LUT(model_G.module.msb, torch.arange(-32, 33, 1), 'MSB', stacks, msb_steps)
Branch2LUT(model_G.module.lsb, torch.arange(0, 4, 1), 'LSB', 0, [1])
