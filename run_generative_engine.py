# Import model
from py_utils.utils import *
import torch
import time

# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import queue
import pickle
import os

# Parser for terminal commands
import argparse
parser = argparse.ArgumentParser(description='Monotonic Groove to Drum Generator')
parser.add_argument('--py2pd_port', type=int, default=1123,
                    help='Port for sending messages from python engine to pd (default = 1123)',
                    required=False)
parser.add_argument('--pd2py_port', type=int, default=1415,
                    help='Port for receiving messages sent from pd within the py program (default = 1415)',
                    required=False)
parser.add_argument('--wait', type=float, default=2,
                    help='minimum rate of wait time (in seconds) between two executive generation (default = 2 seconds)',
                    required=False)
parser.add_argument('--show_count', type=bool, default=True,
                    help='prints out the number of sequences generated',
                    required=False)
# parser.add_argument('--model', type=str, default="100",
#                     help='name of the model: (1) light_version: less computationally intensive, or '
#                          '(2) heavy_version: more computationally intensive',
#                     required=False)

args = parser.parse_args()

if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_path = f"trained_torch_models"
    model_name = "100.pth"
    show_count = args.show_count

    groove_transformer_vae = load_model(model_name, model_path)

    voice_thresholds = [0.01 for _ in range(9)]
    voice_max_count_allowed = [16 for _ in range(9)]

    # load per style means/stds of z encodings
    file = open(os.path.join(model_path, "z_means.pkl"), 'rb')
    z_means_dict = pickle.load(file)
    print(z_means_dict)
    file.close()
    file = open(os.path.join(model_path, "z_stds.pkl"), 'rb')
    z_stds_dict = pickle.load(file)

    print("Available styles: ", list(z_means_dict.keys()))
    # get_random_sample_from_style(style_="global",
    #                              model_=groove_transformer_vae,
    #                              voice_thresholds_=voice_thresholds,
    #                              voice_max_count_allowed_=voice_max_count_allowed,
    #                              z_means_dict=z_means_dict,
    #                              z_stds_dict=z_stds_dict,
    #                              scale_means_factor=1.0, scale_stds_factor=1.0)

    # ------  Create an empty an empty torch tensor
    input_tensor = torch.zeros((1, 32, 3))

    # ------  Create an empty h, v, o tuple for previously generated events to avoid duplicate messages
    (h_old, v_old, o_old) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))
    
    # set the minimum time needed between generations
    min_wait_time_btn_gens = args.wait


    # -----------------------------------------------------

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_pd_port = args.pd2py_port
    sending_to_pd_port = args.py2pd_port
    message_queue = queue.Queue()
    # ----------------------------------------------------------

    # ------------------ OSC Receiver from Pd ------------------ #
    # create an instance of the osc_sender class above
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_pd_port)
    # ---------------------------------------------------------- #


    ## SEND TO PD

    class Z:
        def __init__(self,dl,dr,ul,ur):
            self.lower_left=dl
            self.lower_right=dr
            self.upper_left=ul
            self.upper_right=ur

    class location:
        def __init__(self,x,y):
            self.x=x
            self.y=y


    def send_to_pd(h_new,v_new,o_new):
        osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
        print(osc_messages_to_send)
    

        # First clear generations on pd by sending a message
        py_to_pd_OscSender.send_message("/reset_table", 1)

        # Then send over generated notes one at a time
        for (address, h_v_ix_tuple) in osc_messages_to_send:
            py_to_pd_OscSender.send_message(address, h_v_ix_tuple)


    ## INTERPOLATION

    def bilinear_interpolation(x,y, z_0, z_1, z_2, z_3,
                           corner_0=(0, 0), corner_1=(1, 0), corner_2=(1, 1), corner_3=(0, 1)):
        """
        Bilinear interpolation of a point in a 2D space.

        Args:
            new_location:   A tuple (x, y) of the new location to interpolate.
            z_0:            The embedding corresponding to the bottomn left corner.
            z_1:            The embedding corresponding to the bottom right corner.
            z_2:            The embedding corresponding to the top right corner.
            z_3:            The embedding corresponding to the top left corner.
            corner_0:       The coordinates of the bottom left corner on the x-y plane.
            corner_1:       The coordinates of the bottom right corner on the x-y plane.
            corner_2:       The coordinates of the top right corner on the x-y plane.
            corner_3:       The coordinates of the top left corner on the x-y plane.
        Returns:
        """

        
        x0, y0 = corner_0
        x1, y1 = corner_1
        x2, y2 = corner_2
        x3, y3 = corner_3

        # Calculate the fractional distances in the x and y directions between the new location
        # and the bottom left corner.
        dx = (x - x0) / (x1 - x0)
        dy = (y - y0) / (y3 - y0)

        # Interpolate the values along the x-axis first.
        z_bottom = z_0 * (1 - dx) + z_1 * dx
        z_top = z_3 * (1 - dx) + z_2 * dx

        # Interpolate the values along the y-axis next.
        z_interp = z_bottom * (1 - dy) + z_top * dy

        return z_interp

    def process_message_from_queue(address, args,embeddings):
        if "VelutimeIndex" in address:
            input_tensor[:, int(args[2]), 0] = 1 if args[0] > 0 else 0  # set hit
            input_tensor[:, int(args[2]), 1] = args[0] / 127  # set velocity
            input_tensor[:, int(args[2]), 2] = args[1]  # set utiming
        elif "threshold" in address:
            voice_thresholds[int(address.split("/")[-1])] = 1-args[0]
        elif "max_count" in address:
            voice_max_count_allowed[int(address.split("/")[-1])] = int(args[0])
        elif "regenerate" in address:
            pass
        elif "time_between_generations" in address:
            global min_wait_time_btn_gens
            min_wait_time_btn_gens = args[0]
        elif "lower_left" in address:
            z1,_=get_random_sample_from_style(style_="funk", model_=groove_transformer_vae,
                                 voice_thresholds_=voice_thresholds,
                                 voice_max_count_allowed_=voice_max_count_allowed,
                                 z_means_dict=z_means_dict, z_stds_dict=z_stds_dict,
                                 scale_means_factor=1.0, scale_stds_factor=1.0)
            embeddings.lower_left=z1
            h_lower_left,v_lower_left,v_lower_left= decode_z_into_drums(model_=groove_transformer_vae, latent_z=z1, voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            send_to_pd(h_lower_left,v_lower_left,v_lower_left)
        elif "lower_right" in address:
            z2,_=get_random_sample_from_style(style_="hiphop", model_=groove_transformer_vae,
                                 voice_thresholds_=voice_thresholds,
                                 voice_max_count_allowed_=voice_max_count_allowed,
                                 z_means_dict=z_means_dict, z_stds_dict=z_stds_dict,
                                 scale_means_factor=1.0, scale_stds_factor=1.0)
            embeddings.lower_right=z1
            h_lower_right,v_lower_right,o_lower_right= decode_z_into_drums(model_=groove_transformer_vae, latent_z=z2, voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            send_to_pd(h_lower_right,v_lower_right,o_lower_right)
        elif "upper_left" in address:
            z3,_=get_random_sample_from_style(style_="jazz", model_=groove_transformer_vae,
                                 voice_thresholds_=voice_thresholds,
                                 voice_max_count_allowed_=voice_max_count_allowed,
                                 z_means_dict=z_means_dict, z_stds_dict=z_stds_dict,
                                 scale_means_factor=1.0, scale_stds_factor=1.0)
            embeddings.upper_left=z1
            h_upper_left,v_upper_left,o_upper_left= decode_z_into_drums(model_=groove_transformer_vae, latent_z=z3, voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            send_to_pd(h_upper_left,v_upper_left,o_upper_left)
        elif "upper_right" in address:
            z4,_=get_random_sample_from_style(style_="reggae", model_=groove_transformer_vae,
                                 voice_thresholds_=voice_thresholds,
                                 voice_max_count_allowed_=voice_max_count_allowed,
                                 z_means_dict=z_means_dict, z_stds_dict=z_stds_dict,
                                 scale_means_factor=1.0, scale_stds_factor=1.0)
            embeddings.upper_right=z1
            h_upper_right,v_upper_right,o_upper_right= decode_z_into_drums(model_=groove_transformer_vae, latent_z=z4, voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            send_to_pd(h_upper_right,v_upper_right,o_upper_right)
        elif "x" in address:
            xy.x=valor
        elif "y" in address:
            xy.y=valor
        else:
            print ("Unknown Message Received, address {}, value {}".format(address, args))

    # python-osc method for establishing the UDP communication with pd
    server = OscMessageReceiver(ip, receiving_from_pd_port, message_queue=message_queue)
    server.start()

    embeddings=Z(0,0,0,0)
    xy=location(0,0)
    
    
    interpolation=True
    number_of_generations = 0
    count = 0
    while (1):
        address, args = message_queue.get()
        process_message_from_queue(address, args,embeddings)

        # only generate new pattern when there isnt any other osc messages backed up for processing in the message_queue
        if (message_queue.qsize() == 0) and interpolation==True:

            # ----------------------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------------------- #
            # EITHER GENERATE USING GROOVE OR GENERATE A RANDOM PATTERN

            # case 1. generate using groove
            
            #location=None
            #z_interp=bilinear_interpolation(x=xy.x,y=xy.y, embeddings.lower_left, embeddings.lower_right, embeddings.upper_left, embeddings.upper_right,
            #               corner_0=(0, 0), corner_1=(1, 0), corner_2=(1, 1), corner_3=(0, 1))

            
            
            #h,v,o=decode_z_into_drums(model_=groove_transformer_vae, latent_z=z_interp, voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            
            h,v,o=decode_z_into_drums(model_=groove_transformer_vae, latent_z=np.random.random((128)), voice_thresholds=voice_thresholds, voice_max_count_allowed=voice_max_count_allowed)
            
            # ----------------------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------------------- #
            # send to pd


            osc_messages_to_send = get_new_drum_osc_msgs((h, v, o))
            number_of_generations += 1

            # First clear generations on pd by sending a message
            py_to_pd_OscSender.send_message("/reset_table", 1)

            # Then send over generated notes one at a time
            for (address, h_v_ix_tuple) in osc_messages_to_send:
                py_to_pd_OscSender.send_message(address, h_v_ix_tuple)

            if show_count:
                print("Generation #", count)

            # Message pd that sent is over by sending the counter value for number of generations
            # used so to take snapshots in pd
            py_to_pd_OscSender.send_message("/generation_count", count)

            count += 1

            time.sleep(min_wait_time_btn_gens)