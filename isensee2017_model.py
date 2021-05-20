from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, MaxPooling3D
from keras.engine import Model

from functools import partial

from .unet import create_convolution_block, concatenate
from utils import weighted_dice_coefficient_loss

activation_function = LeakyReLU
rate = (1,1,1)

create_convolution_block = partial(create_convolution_block, activation=activation_function, dilation_rate = rate, instance_normalization=True)

# model taken from 
# https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

def isensee2017(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer="Adam", initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    if optimizer == "YellowFin":
        # import and instantiate yellowFin optimizer
        from .yellowfinkeras.yellowfin import YFOptimizer
        from keras.optimizers import TFOptimizer

        # define the optimizer
        optimizer = TFOptimizer(YFOptimizer())
        model.compile(optimizer=optimizer, loss=loss_function)
    elif optimizer == "Adam":
        from keras.optimizers import Adam
        optimizer = Adam
        model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    else:
        raise ValueError(str(error) + "\n\nYou can use only Adam or YellowFin optimizer\n")
    return model
