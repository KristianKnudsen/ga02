"""
store all the agents here
"""
from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Remains unchanged
class Agent():
    # Remains unchanged

    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):

        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        return row*self._board_size + col


class QNet(nn.Module):
    """
        Description
        ----------
            The model used for prediction and potentially as the target net.
            It will build layers dynamically from the config file.
            It needs to know the bordsize, number of frames and number of actions
            to match the input dimensions of the model. It uses the version number to access the
            configuration file to know which layers to build.

            Attributes
            ------------
            layers : ModuleList
                stores the layers
            dimensions : [int, int, int]
                Keeps track of the input shape for the next layer. In the order [n_frames, boardsize, boardsize]
    """
    def __init__(self, board_size, n_frames, n_actions, version):
        super(QNet, self).__init__()
        self._board_size = board_size
        self._n_frames = n_frames
        self._n_actions = n_actions
        self._version = version

        self.layers = nn.ModuleList()
        self.dimensions = [self._n_frames, self._board_size, self._board_size]
        self.build_layers()

    def build_layers(self):
        """
        Description
        ----------
            Builds the model from the config file specified in training.py. The config file is based on TensorFlow
            naming conventions and may not be directly translateable to PyTorch.
            Therefore we will assume every config file is limited to the same parameters and settings showed in 
            config v17.1.

            We store each layer in the self.layers variable. This is stored in a nn.ModuleList for ease of use and
            compatibility. 

            During building of the model we need to keep track of the dimensions as these are not automatically adjusted
            like in tensorflow. 

            The method is built after config v17.2, there may be missing functionality for activation functions, padding
            and stride if such is used in other configurations.
            
        """
        # Make sure to create a config with a name corresponding to the version defined in trainin.py.
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        # Loops trough every objects within the model key.
        for layer in m['model']:
            # Loops trough all the objects in the model.
            # Checks if the name contains these strings:
            l = m['model'][layer]
            # Convolutional 2d layer.
            if 'Conv2D' in layer:
                """
                Description
                ----------
                    A convolutional 2d layer without any pooling layers.
                    Main purpose is to learn patterns in the incomming images.
                    With the use of different paddings and kernel size its output shape might
                    change with more than just the in / out channels. 
                    Therefore we have special calcultions in place to keep track or if we see
                    a padding = same we make sure the dimensions are not reduced.

                    If the padding is set to same we need the kernel sizes to be odd or our 
                    calculation wont work.
                    
                    The filters keyword is the number of output filters.

                    The kernel size keyword is the windowsize of the moving windows of the 
                    convolutional algorithm. 

                    Stride is always set to 1

                    The dimension output = floor( (in - kernel_size+2*padding)/stride ) + 1
                    
                """
                filters = l["filters"]
                ks = l["kernel_size"]
                # padding same means no change for the input / output dimensions only the channels.
                if l.get("padding", "") == "same":
                    """the forumula for what padding should be given stride = 1 is:
                        padding = (kernel_size - 1) / 2."""
                    padding = ((ks[0] - 1) // 2, (ks[1] - 1) // 2)
                    cl = nn.Conv2d(in_channels=self.dimensions[0],
                                out_channels=filters,
                                kernel_size=ks, padding=padding)
                else:
                    cl = nn.Conv2d(in_channels=self.dimensions[0],
                                out_channels=filters,
                                kernel_size=ks)
                    """If no padding is specified the padding is 0. Then the output dimension will be:
                        dimension = in_dimension - kernel_size +1"""
                    self.dimensions[1] = self.dimensions[1] - ks[0] + 1
                    self.dimensions[2] = self.dimensions[2] - ks[1] + 1
                    
                self.layers.append(cl)
                self.dimensions[0] = filters
                print(self.dimensions)

            # Flatten the shape to a single dimension. [batchsize, 1]
            elif 'Flatten' in layer:
                self.layers.append(nn.Flatten())
                # Multiply the dimensions.
                self.dimensions = [self.dimensions[0] * self.dimensions[1] * self.dimensions[2]]

            # Fully connected linear layer.
            elif 'Dense' in layer:
                # Units are the number of neurons and the same as the number of outputs.
                units = l["units"]
                # dimensions in changed to 'units'
                dl = nn.Linear(in_features=self.dimensions[0], out_features=units)
                self.layers.append(dl)
                self.dimensions[0] = units

            # Inside every object we may call an activation function
            # Currently only ReLu is implemented.
            if l.get("activation", "") == "relu":
                self.layers.append(nn.ReLU())

        # Final linear layer to map the model output to the desired dimension. 
        # No activation function here(important).
        # Allows the outputted Q values to be any range of numbers.
        self.layers.append(nn.Linear(in_features=self.dimensions[0], out_features=self._n_actions))


    def forward(self, x):
        """
            Description
            ----------
                Used to make prediction based on the model.
                Loops trough the layers stored in self.layers and applies them to the input.

            Parameters
            ----------
                x : PyTorch Tensor
                    Board states in the format defined by the config. The order [batchsize, n_frames, boardsize, boardsize]
                    v17.1 config: the input will be tensor([64, 2, 10, 10])

            Returns
            -------
                model output x : PyTorch Tensor
                    Returns the predictions based on the states. The return format will be [batchsize, n_actions]
                    v17.1 config: this will be tensor([64, 4])
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : PyTorch model
        Stores the graph of the DQN model
    _target_net : PyTorch model
        Stores the target network graph of the DQN model
    _optimizer : PyTorch Optimizer
        Stores the optimizer of the _model attribute.
    _criterion : function
        Stores the loss function currently used.
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.95, n_actions=3, use_target_net=True,
                 version=''):
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                         gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                         version=version)
        self.reset_models()

    def _agent_model(self, target=False):
        """
        Description
        ----------
            Used to initilize, create and retrieve the models used in the DQN.
            Sends the parameters to the QNet class to build the model.
            Prints the model to console.

            Also sets the optimizer and criterion attributes.
            Binds the optimizer to the last model created with the target flag set
            to false.

        Parameters
        ----------
            target : bool, optional
                If the model is going to be used for target, don't set optimizer and criterion

        Returns
        -------
            model : PyTorch model
                Instance of a pytorch cnn model
        """
        # Create model
        model = QNet(self._board_size, self._n_frames, self._n_actions, self._version)
        print(model)
        # If target = false.
        if not target:
            # Create and bind optimizer.
            # Change learining rate here.
            self._optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
            # Loss function
            self._criterion = nn.HuberLoss()
        return model

    def _get_model_outputs(self, board, model=None, no_grad = False):
        """
        Description
        ----------
            Used to get the outputs from a model. If no model is provided use the default self._model model.

        Parameters
        ----------
            board: PyTorch Tensor / numpy array
                    Board states in the format defined by the config. The order [batchsize, boardsize, boardsize, n_frames]
                    v17.1 config: the input will be tensor([64, 2, 10, 10])
            model : PyTorch model, optional
                If the model is going to be used for target, don't set optimizer and criterion
            no_grad : bool, optional
                Used to disable gradient calculation. Mostly when using the target network or
                evaluating performance. Stops backpropigation

        Returns
        -------
            predictions : PyTorch Tensor
                Returns Q value preditctions based on the boards provided with the method.
                usually in the format tensor([batchsize, n_actions])
        """

        # By default we use the self._model
        if not model:
            model = self._model

        # This function have been used outside of the agent.py file.
        # When this happens it is usually in numpy format.
        # When working with PyTorch we need to deal with PyTorch Tensors. 
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float()

        # Normalize the board state.
        # Since the board is made of numbers from 0 - 4 dividing by 4
        # keeps all the numbers in a more manageable range.
        board /= 4.0

        # If batchsize is 1 the shape of the input board will be
        # [boardsize, boardsize, n_frames] instead of
        # [batchsize, boardsize, boardsize, n_frames].
        # We're adding the batchsize dimension if this is the case.
        if len(board.shape) == 3:
            board = board.unsqueeze(0)

        # We want the frames to represent the channels when we send it into the model.
        # Since the format is [batchsize, boardsize, boardsize, n_frames] we can swap
        # index 2 and 3 and get the correct format: [batchsize, n_frames, boardsize, boardsize]
        if board.shape[1] != self._n_frames:
            board = board.permute(0, 3, 1, 2)

        # Stops gradient calulation and tracking.
        if no_grad:
            with torch.no_grad():
                return model(board)

        # shape: [batchsize, n_actions]
        return model(board)
    

    def get_action_proba(self, board, values=None):
        """
        Description
        ----------
            Gets the probabilities of each move predicted by the model from the boardstates.
            Does not consider legal moves as the original version doesnt.
            Uses the softmax function to determine the probabilites. 

        Parameters
        ----------
            board: PyTorch Tensor
                    Board states in the format defined by the config. The order [batchsize, n_frames, boardsize, boardsize]
                    v17.1 config: the input will be tensor([64, 2, 10, 10])
            values : None, optional
                Not used. Implemented for compatibility reasons.

        Returns
        -------
            predictions : PyTorch Tensor
                Returns preditctions based on the boards provided with the method as probabilities.
                usually in the format tensor([batchsize, n_actions])
        """
        # Method is not used in the training loop therefore we call no_grad.
        # output should be Q values: tensor([batchsize, n_actions])
        model_output = self._get_model_outputs(board, self._model, no_grad=True)
        # Softmax function on the n_actions dimnesion.
        # output should be probabilites: tensor([batchsize, n_actions])
        # Where the indecies of the probabilites link to the probability of the action.
        action_proba = F.softmax(model_output, dim=1)
        return action_proba
    

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """
        Description
        ----------
            Train the model by sampling from buffer and return the error.
            
            This function implements the Q-learning update rule using the Bellman equation. 
            It calculates the huberloss for updating the model based on the difference between the 
            estimated Q-values (main model) and the target Q-values (target net).
            
            The target Q-values are computed as the immediate reward plus the discounted future rewards.
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the pytorch sign command
            rewards >0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
                huber loss result
        """
        # Retrieve random boardstates from the buffer.
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)


        # All the variables from the buffer are in numpy arrays.
        # We need to work with pytorch tensors so we convert them.

        # This code is built for the last 2 states.
        # We take the first state and record the action and reward from this state
        # and get the second state. 
        # from this second state we need to know the legal moves.
        # If it dies taking the action instead we set the done variable to 1.

        # The first state in the function is refert to as the current state or just state.
        # We call the second state next_states.
        # states. States, boardstates, board and boards are used interchangeably.

        # [batchsize, boarszie, boardsize, n_frames]
        states = torch.from_numpy(s).float()
        # [batchsize, n_actions]
        actions = torch.from_numpy(a).long()
        # [batchsize, 1]
        rewards = torch.from_numpy(r).float()
        # [batchsize, boarszie, boardsize, n_frames]
        next_states = torch.from_numpy(next_s).float()
        # [batchsize, 1]
        done = torch.from_numpy(done).float()
        # [batchsize, n_actions]
        legal_moves = torch.from_numpy(legal_moves)

        # May change the behavior of the model.
        if reward_clip:
            rewards = rewards.sign()

        # Getting the indecies of which the actions are located.
        # ex action = [0, 1, 0, 0], index = 2.
        action_indices = actions.argmax(dim=1)

        # Gather Q-values for the actions taken
        current_q_values = self._get_model_outputs(states, self._model)
        # Gathers the q values of the indecies in action indecies from dimension 1.
        # Basically reduces the current q values to only the actions taken q values.
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1))

        # Gets the target net model prediction.
        next_q_values = self._get_model_outputs(next_states, self._target_net, no_grad=True)
        # Sets illegal moves to -infinity so we dont accidentaly chose this.
        # Ex a snake can not turn 180 degrees
        next_q_values[legal_moves == 0] = float('-inf')
        # Reduces the q values to only the highest ones.
        next_q_values = next_q_values.max(1)[0]

        # [batchsize, 1] -> [batchsize]
        rewards = rewards.squeeze()
        # [batchsize, 1] -> [batchsize]
        done = done.squeeze()
    
        # Calculate the expected q values using the Bellman equation.
        # Since we can get a better approximation one step forward we can use this
        # to adjust the q values one step back.
        # the key here is that we add the actual rewards which are 100% accurate.
        # We then add the future discounted rewards which are approximated, but should still be more
        # accurate.
        expected_q_values = rewards + (1 - done) * self.get_gamma() * next_q_values

        # Huber loss
        loss = self._criterion(current_q_values.squeeze(), expected_q_values)

        # back propagation

        # Prepare gradients.
        self._optimizer.zero_grad()
        # compute gradients
        loss.backward()
        # update parameters
        self._optimizer.step()

        return loss.item()

    def move(self, board, legal_moves, value=None):
        """
        Description
        ----------
            Takes board states and its corresponding legal moves.
            Then makes a prediction on the next best move using the main model.
            This is purley for prediction and not training.
        
        Parameters
        ----------
        board: PyTorch Tensor / numpy array
            Board states in the format defined by the config. The order [batchsize, boardsize, boardsize, n_frames]
            v17.1 config: the input will be tensor([64, 2, 10, 10])
        legal_moves : numpy array
            0 represents illegal moves where 1 represents legal ones. [batchsize, n_actions]
        values : None, optional
            Not used. Implemented for compatibility reasons.

        Returns
        -------
            best_actions : numpy array
                contains the best action indecies. 
        """

        # We want to deal with tensors.
        if isinstance(legal_moves, np.ndarray):
            legal_moves = torch.from_numpy(legal_moves)

        # get model outputs will convert board to a tensor.
        # No gradient tracking.
        q_values = self._get_model_outputs(board, self._model, no_grad=True)

        # illegal moves are set to -ininity to not be chosen.
        q_values[legal_moves == 0] = float('-inf')
        # choose best move.
        best_actions = q_values.max(1)[1]
        # return best move
        return best_actions.numpy()
    

    "Some minor methods with less descriptions"
    #"---------------------------------------------------------------------------------"

    # gets new instances of each model and sets them to their corresponding attributes.
    def reset_models(self):
        self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model(target=True)
            self.update_target_net()

    # Copies the weight from main model to the target net
    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    # Saves the model and target net in a pth file.
    # Iteration is defined by the training loop and is the current episode.
    def save_model(self, file_path='', iteration=None):
        iteration = 0 if iteration is None else iteration
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")

    # loads the model from the files saved using save_model.
    # the models should have similar structure or it wont work.
    def load_model(self, file_path='', iteration=None):
        iteration = 0 if iteration is None else iteration
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pth"))
        if self._use_target_net:
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pth"))