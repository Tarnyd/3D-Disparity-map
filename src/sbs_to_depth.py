import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from defisheye import Defisheye # Import defisheye
from PIL import Image # For conversion from/to PIL if needed

# Global variables to hold image data for interactive callback (simplification)
# In a more complex application, one would use classes or userdata.
g_left_gray_interactive = None
g_right_gray_interactive = None
g_interactive_window_name = "SGBM Interactive Tuning"
g_disparity_display_window = "Disparity Map (Interactive)"

# Container for SGBM parameters used in interactive mode
g_sgbm_params = {}


def undistort_image_defisheye(image_np_bgr, dtype='linear', format='fullframe', fov=190, pfov=120):
    """
    Corrects fisheye distortion on a single image using the Defisheye library.
    """
    try:
        # Defisheye might expect a file path or a PIL Image object.
        # If we have a NumPy array, convert it to a PIL Image.
        # OpenCV reads BGR, PIL uses RGB.
        image_pil_rgb = Image.fromarray(cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB))

        obj = Defisheye(image_pil_rgb, dtype=dtype, format=format, fov=fov, pfov=pfov)
        undistorted_pil_rgb = obj.convert()

        # Convert back to OpenCV NumPy array (BGR)
        undistorted_np_bgr = cv2.cvtColor(np.array(undistorted_pil_rgb), cv2.COLOR_RGB2BGR)
        return undistorted_np_bgr
    except Exception as e:
        print(f"Error during defisheye processing: {e}")
        print("Returning original image.")
        return image_np_bgr

def update_sgbm_disparity_interactive(_=None): # _ receives the trackbar position, but we read all of them
    """
    Callback function for trackbars. Calculates and displays the disparity map.
    """
    if g_left_gray_interactive is None or g_right_gray_interactive is None:
        return

    # Get values from trackbars
    # SGBM-specific parameters
    min_disp = cv2.getTrackbarPos('minDisparity', g_interactive_window_name)
    # numDisparities must be > 0 and divisible by 16
    num_disp_raw = cv2.getTrackbarPos('numDisparities', g_interactive_window_name)
    num_disp = (num_disp_raw // 16) * 16
    if num_disp == 0: num_disp = 16
    cv2.setTrackbarPos('numDisparities', g_interactive_window_name, num_disp) # Update trackbar if value adjusted

    # blockSize must be odd and >= 1
    block_size_raw = cv2.getTrackbarPos('blockSize', g_interactive_window_name)
    block_size = (block_size_raw // 2) * 2 + 1
    if block_size == 0: block_size = 1 # Should be at least 1, will become 1 after (0//2)*2+1
    cv2.setTrackbarPos('blockSize', g_interactive_window_name, block_size)

    P1 = cv2.getTrackbarPos('P1', g_interactive_window_name)
    P2 = cv2.getTrackbarPos('P2', g_interactive_window_name)
    if P2 <= P1 : # Ensure P2 > P1
        P2 = P1 + 1
        cv2.setTrackbarPos('P2', g_interactive_window_name, P2)

    disp12_max_diff = cv2.getTrackbarPos('disp12MaxDiff', g_interactive_window_name)
    pre_filter_cap = cv2.getTrackbarPos('preFilterCap', g_interactive_window_name)
    uniqueness_ratio = cv2.getTrackbarPos('uniquenessRatio', g_interactive_window_name)
    speckle_window_size = cv2.getTrackbarPos('speckleWinSize', g_interactive_window_name)
    speckle_range = cv2.getTrackbarPos('speckleRange', g_interactive_window_name)
    mode_idx = cv2.getTrackbarPos('SGBM Mode', g_interactive_window_name)
    
    sgbm_modes = [cv2.STEREO_SGBM_MODE_SGBM, cv2.STEREO_SGBM_MODE_HH, cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH4]
    mode = sgbm_modes[mode_idx]


    # Update global SGBM parameters (can be printed on exit)
    g_sgbm_params.update({
        'minDisparity': min_disp, 'numDisparities': num_disp, 'blockSize': block_size,
        'P1': P1, 'P2': P2, 'disp12MaxDiff': disp12_max_diff,
        'preFilterCap': pre_filter_cap, 'uniquenessRatio': uniqueness_ratio,
        'speckleWindowSize': speckle_window_size, 'speckleRange': speckle_range,
        'mode': mode_idx # Save mode index
    })

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        preFilterCap=pre_filter_cap,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode
    )

    disparity_map_raw = stereo.compute(g_left_gray_interactive, g_right_gray_interactive)

    if disparity_map_raw is not None:
        disparity_map_float = disparity_map_raw.astype(np.float32) / 16.0
        disparity_map_normalized = cv2.normalize(
            disparity_map_float, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        # Apply a colormap for better visualization
        disparity_colormap = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_JET)
        cv2.imshow(g_disparity_display_window, disparity_colormap)
    else:
        # Create a black image if disparity map is None
        h, w = g_left_gray_interactive.shape
        cv2.imshow(g_disparity_display_window, np.zeros((h, w, 3), dtype=np.uint8))


def run_interactive_sgbm_tuning(left_gray, right_gray):
    """
    Starts an interactive window for tuning SGBM parameters.
    """
    global g_left_gray_interactive, g_right_gray_interactive, g_sgbm_params

    g_left_gray_interactive = left_gray.copy()
    g_right_gray_interactive = right_gray.copy()

    cv2.namedWindow(g_interactive_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(g_disparity_display_window, cv2.WINDOW_NORMAL)
    
    # Show left/right images for reference
    cv2.imshow("Left Gray (Interactive)", g_left_gray_interactive)
    cv2.imshow("Right Gray (Interactive)", g_right_gray_interactive)


    # Initial SGBM parameters and trackbar ranges
    s_temp = cv2.StereoSGBM_create() # Get default values from a fresh object

    cv2.createTrackbar('minDisparity', g_interactive_window_name, 0, 200, update_sgbm_disparity_interactive)
    cv2.createTrackbar('numDisparities', g_interactive_window_name, s_temp.getNumDisparities(), 1024, update_sgbm_disparity_interactive) 
    cv2.createTrackbar('blockSize', g_interactive_window_name, s_temp.getBlockSize(), 51, update_sgbm_disparity_interactive)
    cv2.createTrackbar('P1', g_interactive_window_name, s_temp.getP1(), 3000, update_sgbm_disparity_interactive)
    cv2.createTrackbar('P2', g_interactive_window_name, s_temp.getP2(), 10000, update_sgbm_disparity_interactive)
    cv2.createTrackbar('disp12MaxDiff', g_interactive_window_name, s_temp.getDisp12MaxDiff(), 100, update_sgbm_disparity_interactive)
    cv2.createTrackbar('preFilterCap', g_interactive_window_name, s_temp.getPreFilterCap(), 100, update_sgbm_disparity_interactive)
    cv2.createTrackbar('uniquenessRatio', g_interactive_window_name, s_temp.getUniquenessRatio(), 50, update_sgbm_disparity_interactive)
    cv2.createTrackbar('speckleWinSize', g_interactive_window_name, s_temp.getSpeckleWindowSize(), 200, update_sgbm_disparity_interactive)
    cv2.createTrackbar('speckleRange', g_interactive_window_name, s_temp.getSpeckleRange(), 100, update_sgbm_disparity_interactive)
    cv2.createTrackbar('SGBM Mode', g_interactive_window_name, 2, 3, update_sgbm_disparity_interactive) # 0:SGBM, 1:HH, 2:SGBM_3WAY, 3:HH4

    print("\n--- INTERACTIVE SGBM TUNING ---")
    print("Adjust parameters in the 'SGBM Interactive Tuning' window.")
    print("The result is displayed in 'Disparity Map (Interactive)'.")
    print("Press 'q' in one of the OpenCV windows to exit interactive mode.")
    print("The last used parameters will be printed when you exit.")

    update_sgbm_disparity_interactive() # Show initial disparity map

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
    
    print("\n--- Exited interactive mode ---")
    print("Last used SGBM parameters (for 'mode', index 0-3 is shown):")
    for key, value in g_sgbm_params.items():
        print(f"  {key}: {value}")
    print("You can use these values as arguments if you run the script without --interactive_mode.")
    print("Example: --sgbm_num_disp 128 --sgbm_block_size 5 ... etc.")

    cv2.destroyAllWindows()


def create_disparity_map_from_sbs(sbs_image_path,
                                  algorithm='SGBM',
                                  output_dir='.',
                                  base_output_filename='disparity_map',
                                  defisheye_params=None, # Dictionary
                                  sgbm_overrides=None, # Dictionary for SGBM params
                                  # interactive_scale_factor is handled in main for interactive mode
                                  ):
    sbs_image = cv2.imread(sbs_image_path)
    if sbs_image is None:
        print(f"Error: Could not read image from {sbs_image_path}")
        return None

    height, width, _ = sbs_image.shape
    mid_point = width // 2
    left_image_bgr_orig = sbs_image[:, :mid_point]
    right_image_bgr_orig = sbs_image[:, mid_point:]

    left_image_bgr = left_image_bgr_orig
    right_image_bgr = right_image_bgr_orig

    if defisheye_params and defisheye_params.get('fov', 0) > 0:
        print("Applying Defisheye correction...")
        print(f"  Defisheye parameters: {defisheye_params}")
        left_image_bgr = undistort_image_defisheye(
            left_image_bgr_orig,
            dtype=defisheye_params.get('dtype', 'linear'),
            format=defisheye_params.get('format', 'fullframe'),
            fov=defisheye_params['fov'],
            pfov=defisheye_params.get('pfov', 120)
        )
        right_image_bgr = undistort_image_defisheye(
            right_image_bgr_orig,
            dtype=defisheye_params.get('dtype', 'linear'),
            format=defisheye_params.get('format', 'fullframe'),
            fov=defisheye_params['fov'],
            pfov=defisheye_params.get('pfov', 120)
        )
        print("Defisheye correction complete.")

    left_gray = cv2.cvtColor(left_image_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image_bgr, cv2.COLOR_BGR2GRAY)

    print(f"Image size after potential defisheye (per eye): {left_gray.shape}")

    # --- SGBM Configuration ---
    if algorithm.upper() == 'SGBM':
        s = cv2.StereoSGBM_create() # Get default values
        params = {
            'minDisparity': s.getMinDisparity(),
            'numDisparities': s.getNumDisparities(),
            'blockSize': s.getBlockSize(),
            'P1': s.getP1(),
            'P2': s.getP2(),
            'disp12MaxDiff': s.getDisp12MaxDiff(),
            'preFilterCap': s.getPreFilterCap(),
            'uniquenessRatio': s.getUniquenessRatio(),
            'speckleWindowSize': s.getSpeckleWindowSize(),
            'speckleRange': s.getSpeckleRange(),
            'mode': s.getMode() 
        }
        if sgbm_overrides:
            # Convert mode from index to actual value if it comes from overrides (like from interactive mode)
            if 'mode' in sgbm_overrides and isinstance(sgbm_overrides['mode'], int):
                sgbm_modes = [cv2.STEREO_SGBM_MODE_SGBM, cv2.STEREO_SGBM_MODE_HH, cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH4]
                try:
                    sgbm_overrides['mode'] = sgbm_modes[sgbm_overrides['mode']]
                except IndexError:
                    print(f"Warning: Invalid SGBM mode index {sgbm_overrides['mode']}. Using default.")
                    sgbm_overrides['mode'] = s.getMode()
            
            params.update(sgbm_overrides)
            
            if params['numDisparities'] % 16 != 0 or params['numDisparities'] <= 0:
                params['numDisparities'] = ((params['numDisparities'] // 16) * 16)
                if params['numDisparities'] <= 0 : params['numDisparities'] = 16
                print(f"Adjusted numDisparities to {params['numDisparities']}")
            if params['blockSize'] % 2 == 0 or params['blockSize'] <=0:
                params['blockSize'] = ((params['blockSize'] // 2) * 2) + 1
                if params['blockSize'] <= 0 : params['blockSize'] = 1 # Ensure at least 1
                print(f"Adjusted blockSize to {params['blockSize']}")

        print(f"Using SGBM with parameters: {params}")
        stereo = cv2.StereoSGBM_create(**params)

    elif algorithm.upper() == 'BM':
        s = cv2.StereoBM_create()
        params = {
            'numDisparities': s.getNumDisparities(), 
            'blockSize': s.getBlockSize()          
        }
        # Add BM-specific overrides if needed
        print(f"Using StereoBM with parameters: {params}")
        stereo = cv2.StereoBM_create(**params)
    else:
        print(f"Error: Unknown algorithm '{algorithm}'. Choose 'SGBM' or 'BM'.")
        return None

    print("Calculating disparity map...")
    disparity_map_raw = stereo.compute(left_gray, right_gray)

    if disparity_map_raw is not None:
        disparity_map_float = disparity_map_raw.astype(np.float32) / 16.0
        disparity_map_normalized = cv2.normalize(
            disparity_map_float, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        norm_output_filename = f"{base_output_filename}_{algorithm.lower()}_disparity_normalized.png"
        norm_output_path = os.path.join(output_dir, norm_output_filename)
        try:
            cv2.imwrite(norm_output_path, disparity_map_normalized)
            print(f"Normalized disparity map saved as {norm_output_path}")
        except Exception as e:
            print(f"Error saving {norm_output_path}: {e}")

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(left_image_bgr, cv2.COLOR_BGR2RGB)) 
        plt.title('Left Image (After Defisheye)'); plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(right_image_bgr, cv2.COLOR_BGR2RGB)) 
        plt.title('Right Image (After Defisheye)'); plt.axis('off')
        plt.subplot(1, 3, 3)
        img_disp = plt.imshow(disparity_map_normalized, cmap='jet')
        plt.title('Disparity Map (Normalized)'); plt.colorbar(img_disp); plt.axis('off')
        plt.suptitle(f'Stereo Matching with {algorithm.upper()} for {os.path.basename(sbs_image_path)}', fontsize=16)
        
        plot_filename = f"{base_output_filename}_{algorithm.lower()}_plot_summary.png"
        plot_output_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_output_path)
            print(f"Plot saved as {plot_output_path}")
        except Exception as e:
            print(f"Error saving plot {plot_output_path}: {e}")
        plt.show() 

        return disparity_map_raw
    else:
        print("Error: Could not calculate disparity map.")
        return None

def disparity_to_depth(disparity_map_pixels, focal_length_pixels, baseline_meters):
    depth_map = np.zeros_like(disparity_map_pixels, dtype=np.float32)
    valid_disparity_mask = disparity_map_pixels > 0
    # Depth = (Baseline * Focal Length) / Disparity
    depth_map[valid_disparity_mask] = (focal_length_pixels * baseline_meters) / disparity_map_pixels[valid_disparity_mask]
    return depth_map

def main():
    parser = argparse.ArgumentParser(description="Create a disparity and (optionally) depth map from an SBS image.")
    parser.add_argument("sbs_image_path", type=str, help="Path to the SBS image file.")
    parser.add_argument("--algorithm", type=str, default="SGBM", choices=["SGBM", "BM"], help="Stereo algorithm.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory for output files.")
    
    # Defisheye parameters
    parser.add_argument("--defisheye_fov", type=int, default=0, help="FOV for the fisheye lens (degrees). 0 to not use defisheye. Ex: 190")
    parser.add_argument("--defisheye_pfov", type=int, default=120, help="Preserved FOV after defisheye (degrees). Ex: 120")
    parser.add_argument("--defisheye_dtype", type=str, default="linear", help="Defisheye dtype (e.g., linear, equalarea).")
    parser.add_argument("--defisheye_format", type=str, default="fullframe", help="Defisheye format (e.g., fullframe, circular).")

    # Depth calculation parameters
    parser.add_argument("--focal_length", type=float, default=None, help="Focal length in pixels (for depth calculation).")
    parser.add_argument("--baseline", type=float, default=None, help="Baseline in meters (for depth calculation).")
    parser.add_argument("--max_depth_vis", type=float, default=10.0, help="Max depth for depth map visualization (meters).")

    # Interactive mode
    parser.add_argument("--interactive_mode", action="store_true", help="Start interactive mode for SGBM parameter tuning.")
    parser.add_argument("--interactive_scale", type=float, default=0.25, help="Scale factor for images in interactive mode (e.g., 0.25 for 1/4 size).")

    # SGBM parameter overrides (used if not in interactive_mode)
    parser.add_argument("--sgbm_min_disp", type=int, help="SGBM minDisparity.")
    parser.add_argument("--sgbm_num_disp", type=int, help="SGBM numDisparities.")
    parser.add_argument("--sgbm_block_size", type=int, help="SGBM blockSize.")
    parser.add_argument("--sgbm_p1", type=int, help="SGBM P1.")
    parser.add_argument("--sgbm_p2", type=int, help="SGBM P2.")
    parser.add_argument("--sgbm_disp12_max_diff", type=int, help="SGBM disp12MaxDiff.")
    parser.add_argument("--sgbm_pre_filter_cap", type=int, help="SGBM preFilterCap.")
    parser.add_argument("--sgbm_uniqueness_ratio", type=int, help="SGBM uniquenessRatio.")
    parser.add_argument("--sgbm_speckle_win_size", type=int, help="SGBM speckleWindowSize.")
    parser.add_argument("--sgbm_speckle_range", type=int, help="SGBM speckleRange.")
    parser.add_argument("--sgbm_mode", type=int, help="SGBM mode index (0-3). 0:SGBM, 1:HH, 2:SGBM_3WAY, 3:HH4.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            print(f"Error: Could not create output directory {args.output_dir}: {e}")
            return

    base_filename_for_output = os.path.splitext(os.path.basename(args.sbs_image_path))[0]

    defisheye_args = None
    if args.defisheye_fov > 0:
        defisheye_args = {
            'fov': args.defisheye_fov,
            'pfov': args.defisheye_pfov,
            'dtype': args.defisheye_dtype,
            'format': args.defisheye_format
        }
    
    sgbm_cli_overrides = {
        k.replace('sgbm_', ''): v for k, v in vars(args).items() 
        if k.startswith('sgbm_') and v is not None
    }


    if args.interactive_mode:
        sbs_img_interactive = cv2.imread(args.sbs_image_path)
        if sbs_img_interactive is None:
            print(f"Error: Could not read image for interactive mode: {args.sbs_image_path}")
            return
        
        h_i, w_i, _ = sbs_img_interactive.shape
        mid_i = w_i // 2
        left_bgr_i_orig = sbs_img_interactive[:, :mid_i]
        right_bgr_i_orig = sbs_img_interactive[:, mid_i:]

        if defisheye_args:
            print("Applying Defisheye for interactive mode...")
            left_bgr_i = undistort_image_defisheye(left_bgr_i_orig, **defisheye_args)
            right_bgr_i = undistort_image_defisheye(right_bgr_i_orig, **defisheye_args)
        else:
            left_bgr_i = left_bgr_i_orig
            right_bgr_i = right_bgr_i_orig
            
        left_gray_i = cv2.cvtColor(left_bgr_i, cv2.COLOR_BGR2GRAY)
        right_gray_i = cv2.cvtColor(right_bgr_i, cv2.COLOR_BGR2GRAY)

        if args.interactive_scale != 1.0 and args.interactive_scale > 0:
            print(f"Scaling down images for interactive mode by factor {args.interactive_scale}...")
            new_w = int(left_gray_i.shape[1] * args.interactive_scale)
            new_h = int(left_gray_i.shape[0] * args.interactive_scale)
            if new_w > 0 and new_h > 0 :
                left_gray_i = cv2.resize(left_gray_i, (new_w, new_h), interpolation=cv2.INTER_AREA)
                right_gray_i = cv2.resize(right_gray_i, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"New size for interactive mode (per eye): {left_gray_i.shape}")
            else:
                print("Warning: Scale factor for interactive mode results in 0-dimension. Using original size for interactive mode.")

        run_interactive_sgbm_tuning(left_gray_i, right_gray_i)
        
        print("\nTo run with the last used interactive parameters, use them as command-line arguments, e.g.:")
        # Construct the command string carefully
        cmd_parts = [f"python src/sbs_to_depth.py \"{args.sbs_image_path}\" --output_dir \"{args.output_dir}\""]
        for k, v in g_sgbm_params.items():
            # Need to handle mode index to mode name if desired, or just use index for cli
            param_name_cli = k # For most params, name is same
            if k == 'mode': # SGBM mode is special if we stored index
                param_name_cli = 'sgbm_mode' 
            else:
                param_name_cli = f"sgbm_{k.replace('Disparity', 'Disp').replace('Size','Size').replace('Ratio','Ratio').replace('MaxDiff','MaxDiff').replace('FilterCap','FilterCap').lower()}"
                # A more direct mapping might be better
                # Let's just use the keys as they are stored in g_sgbm_params and ensure CLI args match
                param_name_cli = f"sgbm_{k[0].lower() + k[1:]}" # e.g. sgbm_minDisparity
                # The argparse args are already sgbm_min_disp, sgbm_num_disp etc. So match that.
                arg_name = f"sgbm_{k.replace('D', '_d').replace('B','_b').replace('S','_s').replace('R','_r').replace('M','_m').replace('P','_p').replace('F','_f').replace('C','_c').replace('U','_u').lower()}"
                # This is getting too complex. Let's simplify by making CLI args more direct
                # The current g_sgbm_params keys are 'minDisparity', 'numDisparities', etc.
                # The CLI args are like '--sgbm_min_disp'. Let's just print the map directly for simplicity.
            cmd_parts.append(f"--sgbm_{k.replace('D', '_d').replace('B', '_b').replace('S', '_s').replace('R', '_r').replace('M', '_m').replace('P', '_p').replace('F', '_f').replace('C', '_c').replace('U', '_u').lower().replace('__','_')} {v}")
            # Simpler reconstruction based on actual argparse names:
            # This logic should map g_sgbm_params keys to the argparse dest names.
        
        # Simplified reconstruction based on how argparse names were defined:
        cli_example_parts = [f"python src/sbs_to_depth.py \"{args.sbs_image_path}\" --output_dir \"{args.output_dir}\""]
        arg_map = { # Maps g_sgbm_params keys to CLI option stems
            'minDisparity': 'min_disp', 'numDisparities': 'num_disp', 'blockSize': 'block_size',
            'P1': 'p1', 'P2': 'p2', 'disp12MaxDiff': 'disp12_max_diff',
            'preFilterCap': 'pre_filter_cap', 'uniquenessRatio': 'uniqueness_ratio',
            'speckleWindowSize': 'speckle_win_size', 'speckleRange': 'speckle_range',
            'mode': 'mode' # This is an index
        }
        for g_key, cli_stem in arg_map.items():
            if g_key in g_sgbm_params:
                cli_example_parts.append(f"--sgbm_{cli_stem} {g_sgbm_params[g_key]}")
        
        if defisheye_args:
             cli_example_parts.append(" ".join([f"--defisheye_{k} {v}" for k, v in defisheye_args.items()]))
        print(" ".join(cli_example_parts))


    else: # Normal run (not interactive)
        raw_disparity = create_disparity_map_from_sbs(
            args.sbs_image_path,
            algorithm=args.algorithm,
            output_dir=args.output_dir,
            base_output_filename=base_filename_for_output,
            defisheye_params=defisheye_args,
            sgbm_overrides=sgbm_cli_overrides
        )

        if raw_disparity is not None and args.focal_length is not None and args.baseline is not None:
            print("\nConverting disparity to depth...")
            disparity_pixels = raw_disparity.astype(np.float32) / 16.0
            depth_map_meters = disparity_to_depth(disparity_pixels, args.focal_length, args.baseline)

            depth_map_vis = np.clip(depth_map_meters, 0, args.max_depth_vis)
            depth_map_vis_normalized = cv2.normalize(
                depth_map_vis, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            depth_map_filename = f"{base_filename_for_output}_{args.algorithm.lower()}_depth_normalized.png"
            depth_map_path = os.path.join(args.output_dir, depth_map_filename)
            cv2.imwrite(depth_map_path, depth_map_vis_normalized)
            print(f"Normalized depth map saved as '{depth_map_path}'")

            plt.figure(figsize=(8, 6))
            plt.imshow(depth_map_vis_normalized, cmap='viridis')
            plt.title(f'Depth Map ({args.algorithm.upper()}) for {os.path.basename(args.sbs_image_path)}')
            plt.colorbar(label=f'Depth (normalized, max {args.max_depth_vis}m)')
            plt.axis('off')
            plot_depth_filename = f"{base_filename_for_output}_{args.algorithm.lower()}_depth_plot.png"
            plot_depth_path = os.path.join(args.output_dir, plot_depth_filename)
            plt.savefig(plot_depth_path)
            print(f"Depth map plot saved as {plot_depth_path}")
            plt.show()

        elif (args.focal_length is not None or args.baseline is not None) and \
            not (args.focal_length is not None and args.baseline is not None):
            print("\nWarning: For depth calculation, both --focal_length and --baseline must be provided.")

    print("Processing complete.")

if __name__ == "__main__":
    main()