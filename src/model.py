from keras.models import Model as KerasModel


class Model():
    def get_model(self) -> KerasModel:
        raise NotImplementedError

    def load_model(self) -> KerasModel:
        raise NotImplementedError

    def create_model(self) -> KerasModel:
        raise NotImplementedError

    def get_model_name(self) -> str:
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def predict(self, new_data):
        return self.get_model().predict(new_data)

    def evaluate(self, X_test, y_test):
        return self.get_model().evaluate(X_test, y_test)

    def model_exists(self, name) -> bool:
        return False

    def load_or_create_model(self) -> KerasModel:
        if self.model_exists(self.get_model_name()):
            return self.load_model()
        else:
            return self.create_model()

    def train(
        self,
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_split
    ):
        history = self.get_model().fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )

        return history
