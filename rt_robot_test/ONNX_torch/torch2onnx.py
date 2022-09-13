import sys
# sys.path.append("/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts")

import time
import numpy as np

import torch
import torch.onnx
import torch.nn as nn

import onnx
import onnxruntime

from ACTP.ACTP_model import ACTP
# from ClassifierLSTM.seq_classifier_lstm import ClassifierLSTM
# from CoRL_AC_SP.AC_SP_model import ClassifierLSTM
from CoRL_NONAC_SP.NonAC_SP_model import ClassifierLSTM


device = torch.device("cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    # # This is for the tactile prediction model
    # model_path = "/home/kiyanoush/Cpp_ws/src/robotTest2/torch_models/ACTP"
    # model = torch.load(model_path, map_location="cpu").to(device)
    # model.eval()

    # tactile_dummy_input = torch.rand(10, 1, 48).to(device)
    # action_dummy_input  = torch.rand(20, 1, 2).to(device)
    # t1 = time.time()
    # output              = model.forward(tactile_dummy_input, action_dummy_input)
    # print(time.time() - t1)
  
    # Export the model
    # torch.onnx.export(model,               # model being run
    #                   (tactile_dummy_input, action_dummy_input),  # model input (or a tuple for multiple inputs)
    #                   "pred_model_onnx.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=10,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input1', 'input2'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   dynamic_axes={'input1' : {1 : 'batch_size'},
    #                                 'input2' : {1 : 'batch_size'},    # variable length axes
    #                                 'output' : {1 : 'batch_size'}})
    
    # onnx_model = onnx.load("pred_model_onnx.onnx")
    # onnx.checker.check_model(onnx_model)

    # ort_session = onnxruntime.InferenceSession("pred_model_onnx.onnx")
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tactile_dummy_input), ort_session.get_inputs()[1].name: to_numpy(action_dummy_input)}
    # t1 = time.time()
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(time.time() - t1)
    # # print(ort_outs[0][5, 0, 10])
    # # np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)


    # This is for the slip classification model
    # model_path = "/home/kiyanoush/Cpp_ws/src/robotTest2/torch_models/single_feature_classifier_lstm"
    # model = torch.load(model_path, map_location='cpu').to(device)
    # model.eval()

    # tactile_dummy_input = torch.rand(10, 1, 32).to(device)
    # output              = model.forward(tactile_dummy_input)
  
    # # Export the model
    # torch.onnx.export(model,               # model being run
    #                   (tactile_dummy_input),  # model input (or a tuple for multiple inputs)
    #                   "classification_model_onnx.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=10,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   dynamic_axes={'input' : {1 : 'batch_size'},    # variable length axes
    #                                 'output' : {1 : 'batch_size'}})

    # This is for the CoRL action_conditioned slip prediction model
    # model_path = "/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts/CoRL/torch_models/CoRL_AC_LSTM"
    # model = torch.load(model_path, map_location="cpu").to(device)
    # model.eval()
    # # print(model)

    # tactile_dummy_input = torch.rand(10, 1, 32).to(device)
    # action_dummy_input  = torch.rand(1, 20).to(device)
    # t1 = time.time()
    # output              = model.forward(tactiles=tactile_dummy_input, actions=action_dummy_input)
    # print(time.time() - t1)

    # # Export the model
    # # torch.onnx.export(model,               # model being run
    # #                   (tactile_dummy_input, action_dummy_input),  # model input (or a tuple for multiple inputs)
    # #                   "CoRL_AC_SP.onnx",   # where to save the model (can be a file or file-like object)
    # #                   export_params=True,        # store the trained parameter weights inside the model file
    # #                   opset_version=10,          # the ONNX version to export the model to
    # #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    # #                   input_names = ['input1', 'input2'],   # the model's input names
    # #                   output_names = ['output'], # the model's output names
    # #                   dynamic_axes={'input1' : {1 : 'batch_size'},
    # #                                 'input2' : {0 : 'batch_size'},    # variable length axes
    # #                                 'output' : {0 : 'batch_size'}})

    # onnx_model = onnx.load("CoRL_AC_SP.onnx")
    # onnx.checker.check_model(onnx_model)

    # ort_session = onnxruntime.InferenceSession("CoRL_AC_SP.onnx")
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tactile_dummy_input), ort_session.get_inputs()[1].name: to_numpy(action_dummy_input)}
    # t1 = time.time()
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(time.time() - t1)
    # np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)

    # This is for the CoRL Non_action_conditioned slip prediction model
    model_path = "/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts/CoRL/torch_models/CoRL_NonAC_LSTM"
    model = torch.load(model_path, map_location="cpu").to(device)
    model.eval()
    # print(model)

    tactile_dummy_input = torch.rand(10, 1, 32).to(device)
    t1 = time.time()
    output              = model.forward(tactiles=tactile_dummy_input)
    print(time.time() - t1)

    # Export the model
    # torch.onnx.export(model,               # model being run
    #                   (tactile_dummy_input),  # model input (or a tuple for multiple inputs)
    #                   "CoRL_NonAC_SP.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=10,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input1'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   dynamic_axes={'input1' : {1 : 'batch_size'},
    #                                 'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load("CoRL_NonAC_SP.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("CoRL_NonAC_SP.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tactile_dummy_input)}
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print(time.time() - t1)
    # np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
   

if __name__ == "__main__":
    main()