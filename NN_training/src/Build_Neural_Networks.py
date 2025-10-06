import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_ANN_5_no_BN(nn.Module):
    def __init__(self,n_in, n_out, neurons = 128, dropoff=0.0):
        super(Net_ANN_5_no_BN, self).__init__()
        self.linear1 = nn.Linear(n_in, neurons)
        self.linear2 = nn.Linear(neurons, neurons)
        self.linear3 = nn.Linear(neurons, neurons)
        self.linear4 = nn.Linear(neurons, neurons)
        self.linear5 = nn.Linear(neurons, n_out)

        self.lin_drop = nn.Dropout(dropoff)  # regularization method to prevent overfitting.
        self.lin_drop2 = nn.Dropout(dropoff)
        self.lin_drop3 = nn.Dropout(dropoff)
        self.lin_drop4 = nn.Dropout(dropoff)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.lin_drop(x)
        x = F.relu(self.linear2(x))
        x = self.lin_drop2(x)
        x = F.relu(self.linear3(x))
        x = self.lin_drop3(x)
        x = F.relu(self.linear4(x))
        x = self.lin_drop4(x)
        x = self.linear5(x)
        return x


class CustomNet(nn.Module):
    def __init__(self, layer_sizes, activation_fn=nn.ReLU(), dropoff=0.0):
        """
        Args:
        layer_sizes (list): A list containing the size of each layer. The first element is the input size,
                            and the last element is the output size.
        activation_fn (nn.Module): The activation function to use (default: ReLU).
        dropoff (float or list): Dropout rate applied after each layer. If a list is provided, it must match the number of layers.
        """
        super(CustomNet, self).__init__()
        
        # Creating a list of linear layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
        self.layers = nn.ModuleList(layers)
        
        # Setting the activation function
        self.activation_fn = activation_fn
        
        # If a single dropout rate is given, apply it to all layers, otherwise use the provided list of dropout rates
        if isinstance(dropoff, (int, float)):
            self.dropouts = nn.ModuleList([nn.Dropout(dropoff) for _ in range(len(layer_sizes) - 2)])
        else:
            assert len(dropoff) == len(layer_sizes) - 2, "The length of dropoff list must match the number of layers"
            self.dropouts = nn.ModuleList([nn.Dropout(d) for d in dropoff])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):  # For all layers except the last one
            x = layer(x)
            x = self.activation_fn(x)
            if i < len(self.dropouts):  # Apply dropout if it's not the last layer
                x = self.dropouts[i](x)
                
        x = self.layers[-1](x)  # The final output layer (no activation)
        return x


class ConvColumnNet1D(nn.Module):
    def __init__(self, 
                 conv_layer_configs,  # list of dicts: [{'filters': X, 'kernel': Y}, ...]
                 fc_layer_sizes,
                 activation_fn=nn.ReLU(),
                 dropoff=0.0,
                 levels=49):
        """
        A multi-layer 1D conv + dense model for vertical column prediction.

        Args:
            conv_layer_configs (list): List of dicts for each conv layer (1D conv):
                [{'filters': int, 'kernel': int}, ...]
            fc_layer_sizes (list): Fully connected layer sizes.
            activation_fn (nn.Module): Activation function.
            dropoff (float): Dropout rate.
            levels (int): Number of vertical levels (default 49).
        """
        super(ConvColumnNet1D, self).__init__()

        self.levels = levels
        self.activation_fn = activation_fn

        # Build separate conv stacks for temp and qin
        self.temp_conv_layers = self._build_conv_stack(conv_layer_configs, in_channels=1, dropoff=dropoff)
        self.qin_conv_layers  = self._build_conv_stack(conv_layer_configs, in_channels=1, dropoff=dropoff)

        # Calculate the flattened feature size after conv layers
        final_filters = conv_layer_configs[-1]['filters']
        fc_input_dim = final_filters * levels * 2 + 2  # temp + qin + 2 scalars

        layers = []
        fc_sizes = [fc_input_dim] + fc_layer_sizes
        for i in range(len(fc_sizes) - 1):
            layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            if i < len(fc_sizes) - 2:
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropoff))
        self.fc_layers = nn.Sequential(*layers)

        self.layers = nn.ModuleList([layer for layer in self.fc_layers if isinstance(layer, nn.Linear)])

    def _build_conv_stack(self, conv_configs, in_channels, dropoff):
        layers = []
        for cfg in conv_configs:
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg['filters'],
                kernel_size=cfg['kernel'],
                padding=cfg['kernel'] // 2
            ))
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropoff))
            in_channels = cfg['filters']
        return nn.Sequential(*layers)

    def forward(self, x):
        temp = x[:, :self.levels].unsqueeze(1)                 # (batch, 1, levels)
        qin  = x[:, self.levels:2*self.levels].unsqueeze(1)   # (batch, 1, levels)
        scalars = x[:, 2*self.levels:]                        # (batch, 2)

        temp_feat = self.temp_conv_layers(temp)               # (batch, filters, levels)
        qin_feat  = self.qin_conv_layers(qin)                # (batch, filters, levels)

        temp_flat = temp_feat.view(temp_feat.size(0), -1)     # flatten
        qin_flat  = qin_feat.view(qin_feat.size(0), -1)

        x_combined = torch.cat([temp_flat, qin_flat, scalars], dim=1)
        return self.fc_layers(x_combined)


class ConvColumnNet2D(nn.Module):
    def __init__(self, 
                 conv_layer_configs,  # list of dicts: [{'filters': X, 'kernel': (Y, Z)}, ...]
                 fc_layer_sizes,
                 activation_fn=nn.ReLU(),
                 dropoff=0.0,
                 levels=49):
        """
        A multi-layer 2D conv + dense model for vertical column prediction.
        
        Args:
            conv_layer_configs (list): List of dicts defining each conv layer:
                [{'filters': int, 'kernel': (int, int)}, ...]
            fc_layer_sizes (list): Fully connected layer sizes.
            activation_fn (nn.Module): Activation function.
            dropoff (float): Dropout rate.
            levels (int): Number of vertical levels (default 49).
        """
        super(ConvColumnNet2D, self).__init__()
        self.levels = levels
        self.activation_fn = activation_fn

        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # single input channel: [temp, qin] stacked
        for cfg in conv_layer_configs:
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=cfg['filters'],
                kernel_size=cfg['kernel'],
                padding=(cfg['kernel'][0] // 2, cfg['kernel'][1] // 2)
            ))
            conv_layers.append(activation_fn)
            conv_layers.append(nn.Dropout(dropoff))
            in_channels = cfg['filters']
        self.conv_layers = nn.Sequential(*conv_layers)

        # Dynamically calculate flattened size after conv
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 2, levels)  # (batch, 1, 2, levels)
            conv_out = self.conv_layers(dummy_input)
            flattened_size = conv_out.view(1, -1).size(1)

        fc_input_dim = flattened_size + 2  # +2 for scalar inputs

        # Build fully connected layers
        layers = []
        fc_sizes = [fc_input_dim] + fc_layer_sizes
        for i in range(len(fc_sizes) - 1):
            layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            if i < len(fc_sizes) - 2:
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropoff))
        self.fc_layers = nn.Sequential(*layers)

        # Keep linear layers for saving/exporting
        self.layers = nn.ModuleList([layer for layer in self.fc_layers if isinstance(layer, nn.Linear)])

    def forward(self, x):
        temp_qin = x[:, :2 * self.levels].view(-1, 1, 2, self.levels)  # (batch, 1, 2, levels)
        scalars = x[:, 2 * self.levels:]  # (batch, 2)

        conv_out = self.conv_layers(temp_qin)  # (batch, filters, ?, ?)
        conv_flat = conv_out.view(conv_out.size(0), -1)  # flatten all conv dims

        x_combined = torch.cat([conv_flat, scalars], dim=1)
        return self.fc_layers(x_combined)
