import torch, torchvision, struct, random, threading, gc, os, time
import argparse
from torchvision import transforms
from torchvision.models.alexnet import AlexNet_Weights, alexnet
from torchvision.models.vgg import VGG16_Weights, vgg16
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.models.densenet import DenseNet161_Weights, densenet161
from torchvision.models.inception import Inception_V3_Weights, inception_v3


all_models = {'alexnet': (alexnet, AlexNet_Weights, 56.66, 79.78), 
                'vgg': (vgg16, VGG16_Weights, 72.22, 90.90),
                'resnet': (resnet50, ResNet50_Weights, 80.26, 94.94), 
                'dense': (densenet161, DenseNet161_Weights, 77.20, 93.58),
                'inception': (inception_v3, Inception_V3_Weights, 69.92, 88.74)}

all_flips = {'D1': (81, 14, 8002), 'D3': (61, 14, 17070), 'B1': (96, 0, 12345), 'B2': (18, 6, 42)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_imagenet_models",
        description="Executes bit-flip attack on ImageNet Model workloads, either simulated or time-multiplexed real attack scenario",
    )

    parser.add_argument(
        "model",
        type=str,
        choices=list(all_models.keys()),
        help="Runs the models from the given choices.",
    )

    parser.add_argument(
        "x_mode",
        type=str,
        choices=['sim', 'att'],
        help="Run models in simulation mode or attack mode. Simulation will manually flip in the target bit & byte position. "
        "Attack mode executes model workload in time-multiplexed manner controlled by STDOUT."
    )

    parser.add_argument(
        "flip",
        type=str,
        choices=list(all_flips.keys()),
        help="Choose a bitflip from the choices. Ignored in attack mode."
    )

    parser.add_argument(
        "val_path",
        type=str,
        help="Path to the validation set of the ImageNet models."
    )

    parser.add_argument(
        "alloc_path",
        type=str,
        help="Path to the custom memory allocator. Used only in attack mode."
    )

    parser.add_argument(
        "outfile",
        type=str,
        help="File to store the Maximum RAD parameter"
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threaded instances of the model to run in parallel. Must be less than and divisible by 50. Ignored in attack mode.",
        default=1
    )

    parser.add_argument(
        "--rerun_clean",
        action='store_true',
        help="Do you want to rerun the models to get the clean accuracies?"
    )

    args = parser.parse_args()

    if (args.x_mode == "att"):
        new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            args.alloc_path, 'my_malloc', 'my_free')

        # Swap the current allocator
        torch.cuda.change_current_allocator(new_alloc)

    # 81 61 96 18
    model_control_name = 'model_control.txt'
    exploit_control_name = 'exploit_control.txt'
    target_model, target_weight, clean_top1_acc, clean_top5_acc = all_models[args.model]
    byte_pos, bit_pos, seed = all_flips[args.flip]
    SP = 50
    THREAD_COUNT = args.threads
    BATCH_SIZE = SP // THREAD_COUNT

    # Convert float to IEEE 754 representation as an integer (32-bit)
    def float_to_int(val):
        # Pack the float as IEEE 754 32-bit binary
        packed = struct.pack('!e', val)
        # Convert the packed bytes into an integer
        int_rep = struct.unpack('!H', packed)[0]
        return int_rep

    # Convert IEEE 754 integer representation back to float
    def int_to_float(int_rep):
        # Convert the integer back to bytes
        packed = struct.pack('!H', int_rep)
        # Unpack the bytes as a float
        return struct.unpack('!e', packed)[0]

    # Function to compute Top-1 and Top-5 accuracy
    def compute_accuracy(model, data_loader, device):
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            for images, labels in data_loader:
                # labels = labels.to(torch.bfloat16)
                # images = images.half()
                images = images.to(torch.float16)
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Get top-5 predictions
                _, top5_preds = outputs.topk(5, dim=1)  # Get top-5 indices
                top1_preds = top5_preds[:, 0]  # Top-1 predictions are the first in top-5
                
                # Top-1 accuracy
                top1_correct += (top1_preds == labels).sum().item()
                
                # Top-5 accuracy
                top5_correct += torch.sum(top5_preds.eq(labels.view(-1, 1))).item()
                
                total_samples += labels.size(0)

        top1_accuracy = 100 * top1_correct / total_samples
        top5_accuracy = 100 * top5_correct / total_samples
        return top1_accuracy, top5_accuracy

    def flip_leftmost_bit_32(param, i, bit_to_flip):
        """
        Flip the leftmost (most significant) bit of the first element in a 32-bit PyTorch parameter.
        
        Parameters:
            param (torch.nn.Parameter): The parameter tensor to modify.
        
        Returns:
            original_value (int): The original value before flipping the bit, for restoration.
        """
        # Ensure param is detached from the computation graph to allow modifications
        param = param.detach()

        # Convert the first element to NumPy for bit manipulation, ensuring it's in the int32 format
        param_numpy = param.cpu().float().numpy()  # Convert to NumPy (CPU)
        param_flat = param_numpy.ravel()  # Flatten the tensor
        
        # Get the first element (ensure it's treated as a 32-bit integer)
        first_value = param_flat[i]
        original_value = first_value

        first_value = float_to_int(first_value)
        leftmost_bit_mask = (1 << bit_to_flip)
        # leftmost_bit_mask = ~(1 << 29)
        flipped_value = first_value | leftmost_bit_mask
        param_flat[i] = int_to_float(flipped_value)
        
        # Copy the modified data back into the PyTorch tensor
        param.copy_(torch.from_numpy(param_numpy))

        # Return the original value so it can be used to restore later
        return original_value


    def restore_original_value(param, i, original_value):
        """
        Restore the original value of the parameter to the state before the bit was flipped.
        
        Parameters:
            param (torch.nn.Parameter): The parameter tensor to modify.
            original_value (int): The value before the bit flip to restore.
        """
        # Ensure param is detached from the computation graph to allow modifications
        param = param.detach()

        # Convert the first element to NumPy to modify the value
        param_numpy = param.cpu().numpy()  # Convert to NumPy (CPU)
        param_flat = param_numpy.ravel()  # Flatten the tensor
        
        # Restore the original value
        param_flat[i] = original_value
        
        # Copy the modified data back into the PyTorch tensor
        param.copy_(torch.from_numpy(param_numpy))

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256
        transforms.CenterCrop(224),  # Crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    def validation_thread(weights, parameters_index, clean_top1_acc, clean_top5_acc, subset_s, RAD_counts, stats):
        model = target_model(weights=weights)
        model.eval()
        model.to(torch.float16)
        model.to(device)
        parameters = list(model.parameters())
        print(parameters[0].data_ptr())
        i = 0
        for param_sublist_id, param_id in parameters_index[subset_s * BATCH_SIZE : (subset_s + 1) * BATCH_SIZE]:
            print("Parameter:", param_sublist_id, param_id)
            param = parameters[param_sublist_id]
            param_sublist = param.flatten()
            print("Before flipping the bit:", param_sublist[param_id])
            original = flip_leftmost_bit_32(param, param_id, bit_pos)  # Flip the first bit
            print("After flipping the bit:", param_sublist[param_id])


            # Compute and print Top-1 and Top-5 accuracy
            top1_acc, top5_acc = compute_accuracy(model, data_loader, device)
            top1_RAD = ((clean_top1_acc - top1_acc) / clean_top1_acc)
            top5_RAD = ((clean_top5_acc - top5_acc) / clean_top5_acc)
            print(f"Top-1 Accuracy: {top1_acc:.2f}%")
            print(f"Top-5 Accuracy: {top5_acc:.2f}%")
            print(f"RAD: {top1_RAD}")
            print(f"RAD: {top5_RAD}")
            if top1_RAD > 0.1:
                RAD_counts[subset_s] += 1
            stats[subset_s].append((param_sublist_id, param_id, top1_acc, top5_acc, top1_RAD))
            i = i + 1
            restore_original_value(param, param_id, original)

    imagenet_data = torchvision.datasets.ImageFolder(args.val_path + "/val", transform=val_transforms)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=4,
                                            shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = target_weight.DEFAULT
    model = target_model(weights=weights)
    model.to(torch.float16)
    model.to(device)
    model.eval()

    if (args.rerun_clean):
        print("Retrieving Clean Accuracies...")
        clean_top1_acc, clean_top5_acc = compute_accuracy(model, data_loader, device)
    print(f"Clean Top-1 Accuracy: {clean_top1_acc:.2f}%")
    print(f"Clean Top-5 Accuracy: {clean_top5_acc:.2f}%")
    originalTime = os.path.getmtime(model_control_name)
    with open(exploit_control_name, 'w+') as control:
        control.write("hammer\n")

    if args.x_mode == "att":
        print("Waiting for hammer to complete...")
        while(True):
            if(os.path.getmtime(model_control_name) > originalTime):
                break
            time.sleep(0.1)

        print("Continue...")
        t1, t5 = compute_accuracy(model, data_loader, device)
        top1_RAD = ((clean_top1_acc - t1) / clean_top1_acc)
        top5_RAD = ((clean_top5_acc - t5) / clean_top5_acc)
        print(f"Top-1 Accuracy: {t1:.2f}%")
        print(f"Top-5 Accuracy: {t5:.2f}%")
        print(f"RAD: {top1_RAD}")
        print(f"RAD: {top5_RAD}")

        with open(args.outfile, 'a') as of:
            of.write(f"{t1:.2f},{t5:.2f},{top1_RAD}\n")
        with open(exploit_control_name, 'w+') as control:
            control.write("exit\n")
    else:
        parameters = list(model.parameters())
        parameters_index = set()
        print("Running Model...")

        random.seed(seed)

        next_target = byte_pos
        it = 0
        for i, p in enumerate(parameters):
            for j in range(len(p.flatten())):
                if it == next_target:
                    parameters_index.add((i, j))
                    next_target = next_target + 128
                it += 1
        parameters_index = list(parameters_index)
        parameters_index = random.sample(parameters_index, SP)
        print(parameters_index)

        del model
        gc.collect()
        torch.cuda.empty_cache() 

        threads = []
        RAD_counts = [0 for _ in range(THREAD_COUNT)]
        stats = [[] for _ in range(THREAD_COUNT)]
        for i in range(THREAD_COUNT):
            thread = threading.Thread(target=validation_thread, args=(weights, parameters_index, clean_top1_acc, clean_top5_acc, len(threads), RAD_counts, stats))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        RAD_count = sum(RAD_counts)

        f = open(args.outfile, "w")
        flat_stats = [
            x
            for xs in stats
            for x in xs
        ]

        max_RAD = (-1,-1,-1,-1,-1)
        for tup in flat_stats:
            if tup[4] > max_RAD[4]:
                max_RAD = tup

        for tup in flat_stats:
            print(tup)
        #     for val in tup:
        #         f.write(str(val) + " ") 
        #     f.write("\n")
        # f.close()
        print(f"Vulnerable parameters: {RAD_count}")
        print(f"Vulnerable ratio: {RAD_count / SP}")
        print(max_RAD)
        for val in tup:
            f.write(str(val) + " ")
        f.close()