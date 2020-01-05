from model import Model
from keras.models import Sequential
from keras.layers import Dense


class NeuralNetwork(Model):
    def __init__(
        self,
        input_nodes: int,
        output_nodes: int,
        hidden_layers: int,
        hidden_layer_nodes: int
    ) -> None:
        self.input_nodes: int = input_nodes
        self.output_nodes: int = output_nodes
        self.hidden_layers: int = hidden_layers
        self.hidden_layer_nodes: int = hidden_layer_nodes
        self.name: str = self.get_model_name()
        self.model: Sequential = self.load_or_create_model()

    def create_model(self) -> Sequential:
        model = Sequential()

        input_shape = (self.input_nodes,)
        model.add(
            Dense(
                self.hidden_layer_nodes,
                activation='relu',
                input_shape=input_shape
            )
        )

        for a in range(self.hidden_layers):
            model.add(Dense(self.hidden_layer_nodes))

        model.add(Dense(self.output_nodes))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_model(self):
        return self.model

    def get_model_name(self):
        base = "neural_net"
        inp = "_e" + str(self.input_nodes)
        out = "_t" + str(self.output_nodes)
        hl = "_hl" + str(self.hidden_layers)
        hln = "_hln" + str(self.hidden_layer_nodes)

        return base + inp + out + hl + hln
