import os
import time
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.io
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Sequential, Linear, ReLU
from torchvision.transforms import v2

TRAIN_FOLDER = "/kaggle/input/unibuc-dhc-2023/train_images"
VAL_FOLDER = "/kaggle/input/unibuc-dhc-2023/val_images"
TEST_FOLDER = "/kaggle/input/unibuc-dhc-2023/test_images"

TRAIN_LABELS = "/kaggle/input/unibuc-dhc-2023/train.csv"
VAL_LABELS = "/kaggle/input/unibuc-dhc-2023/val.csv"

TRAIN_IMAGES_NAMES = os.listdir(TRAIN_FOLDER)
VAL_IMAGES_NAMES = os.listdir(VAL_FOLDER)
TEST_IMAGES_NAMES = os.listdir(TEST_FOLDER)


class DeepHallucinationDataSet(torch.utils.data.Dataset):
    """Am extins clasa DataSet din torch pentru a citi si accesa datele.
    Constructorul primeste numele folderului cu imagini, numele imaginilor (in ordinea din folder),
    numele fisierului cu labeluri (pentru train si validare), daca modul curent este train si un parametru
    care ne permite sa transformam datele.
    Dupa analiza datelor, am decis ca singura transformare sa fie un flip random orizontal.
    """

    def __init__(self, folder_name, images_names, label_file_name=None, mode='train', transform=False):
        self.folder = folder_name
        self.images_names = images_names
        if label_file_name is not None:
            self.label_file_name = label_file_name
            self.labels = DeepHallucinationDataSet.read_labels(label_file_name)
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        image = torchvision.io.read_image(os.path.join(self.folder, self.images_names[index]))
        image = torchvision.transforms.functional.convert_image_dtype(image, dtype=torch.float32)
        if self.mode == 'train':
            if self.transform:
                transform_seq = v2.Compose([v2.RandomHorizontalFlip(p=0.5) # se afla intr-un Compose
                                            # deoarece initial am pus mai multe transformari
                                            ])
                image = transform_seq(image)
            label = self.labels[self.images_names[index]]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.images_names)

    @staticmethod
    def read_labels(file_name):
        labels_from_file = pd.read_csv(file_name)
        name_to_class = {image_name: int(class_index) for image_name,
                                                          class_index in zip(labels_from_file["Image"],
                                                                             labels_from_file["Class"])}
        return name_to_class


#####


def write_results(file_name, images_names, pred_test, sample="/kaggle/input/unibuc-dhc-2023/sample_submission.csv"):
    """Functie care primeste numele fisierului unde sa salveze predictiile,
    numele imaginilor in ordinea din folder si predictiile pe datele de test
    Ultimul parametru (sample) a fost necesar pentru a putea salva pentru unele modele si
    predictiile pe setul de validare si aveam nevoie sa stiu ordinea imaginilor din acel csv."""
    res = {im: pred for im, pred in zip(images_names, pred_test)}
    name_order = pd.read_csv(sample)['Image'].tolist()
    ordered_results = [res[img] for img in name_order]
    results = {"Image": name_order, "Class": ordered_results}
    df = pd.DataFrame(data=results)
    df.to_csv(f"/kaggle/working/{file_name}_{time.time()}_results.csv", encoding='utf-8', index=False)


# #### Modelul propus ###


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResNetBlock(torch.nn.Module):
    """Defineste un bloc (mai degraba un subbloc - bloc rezidual) pentru modelul propus de mine.
    Desi definit cu 1 strat convolutional 1x1, un strat convolutional 3x3 si apoi alt  strat convolutional 1x1,
    Primul strat convolutional nu este utilizat.
    Daca are loc schimbarea dimensiunii in bloc, trebuie setat True parametrul change_input, care va trece
    inputul printr-o convolutie de (3,3) cu stride-ul dat pentru intregul subbloc si cu padding 1, pentru a putea fi adaugat
    la rezultatul obtinut prin trecerea sa prin subbloc."""
    def __init__(self, channels_in1, channels_in2, channels_out, stride, change_input=False):
        super(ResNetBlock, self).__init__()
        self.change_input = change_input
        self.convolutional_layer_1 = Conv2d(in_channels=channels_in1,
                                            out_channels=channels_in2,
                                            kernel_size=1,
                                            stride=1,
                                            )  # strat neutilizat
        self.batch_norm_1 = BatchNorm2d(channels_in2)
        self.convolutional_layer_2 = Conv2d(in_channels=channels_in2,
                                            out_channels=channels_in2,
                                            kernel_size=(3, 3),
                                            stride=stride,
                                            padding=1)
        self.batch_norm_2 = BatchNorm2d(channels_in2)
        self.convolutional_layer_3 = Conv2d(in_channels=channels_in2,
                                            out_channels=channels_out,
                                            kernel_size=1,
                                            stride=1,
                                            )
        self.batch_norm_3 = BatchNorm2d(channels_out)
        self.relu = ReLU(inplace=True)
        self.convolutional_layer_4 = None
        if change_input:
            self.convolutional_layer_4 = Sequential(Conv2d(in_channels=channels_in1,
                                                           out_channels=channels_out,
                                                           kernel_size=3,
                                                           stride=stride,
                                                           padding=1
                                                           ), BatchNorm2d(channels_out))

    def forward(self, x):
        # result = self.relu(self.batch_norm_1(self.convolutional_layer_1(x)))
        # cand incercam diverse configuratii am decis sa renunt la primul strat convolutional 1x1
        result = self.relu(self.batch_norm_2(self.convolutional_layer_2(x)))
        result = self.batch_norm_3(self.convolutional_layer_3(result))

        if self.change_input:
            result += self.convolutional_layer_4(x)
        else:
            result += x
        return self.relu(result)

    @staticmethod
    def blocks(c_in, c_in2, c_out, stride, no_blocks=24, change_input=False):
        """Metoda statica care primeste trei dimensiuni si construieste o secventa de subblocuri astfel:
        primul subbloc are stride-ul si i se va aduna inputul modificat, urmatoarele subblocuri sunt cu stride 1
        si inputul nemodifcat."""
        return [ResNetBlock(c_in, c_in2, c_out, stride, change_input)] + \
               [ResNetBlock(c_out, c_out, c_out, 1) for _ in range(no_blocks - 1)]


class ResCNN(torch.nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()

        self.block_1 = Sequential(
            Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_2 = Sequential(*ResNetBlock.blocks(64, 64, 64, 1, 2, True))
        self.block_3 = Sequential(*ResNetBlock.blocks(64, 64, 256, 2, 4, True))
        self.block_4 = Sequential(*ResNetBlock.blocks(256, 256, 512, 2, 1, True))

        self.fully_connected = Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.2),
            Linear(512, 96, device=DEVICE),
        )

        self.block_1.apply(initialize_weights_layer)
        self.block_2.apply(initialize_weights_layer)
        self.block_3.apply(initialize_weights_layer)
        self.block_4.apply(initialize_weights_layer)
        self.fully_connected.apply(initialize_weights_layer)

    def forward(self, x):
        result = self.block_1(x)
        result = self.block_2(result)
        result = self.block_3(result)
        result = self.block_4(result)
        return self.fully_connected(result)


def initialize_weights_layer(layer):
    if type(layer) == Conv2d or type(layer) == Linear:
        torch.nn.init.xavier_normal_(layer.weight, gain=0.5)


if __name__ == '__main__':
    # DataLoading
    train_data = DeepHallucinationDataSet(TRAIN_FOLDER, TRAIN_IMAGES_NAMES, TRAIN_LABELS, transform=True)
    val_data = DeepHallucinationDataSet(VAL_FOLDER, VAL_IMAGES_NAMES, VAL_LABELS)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128)

    # configurarea modelului, a functiei de pierdere, a optimizatorului si a scheduler-ului pentru
    # reducerea ratei de invatare
    my_model = ResCNN().to(DEVICE)
    cross_entropy_loss = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizing_algo = torch.optim.SGD(my_model.parameters(), momentum=0.94, lr=1e-2, weight_decay=0.001)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizing_algo, 'min',
                                                           verbose=True,
                                                           patience=3,
                                                           threshold=0.003,
                                                           factor=0.1)

    # antrenarea modelului
    # pentru antrenare -> eu am antrenat 30 epoci, apoi pana la 50 epoci, apoi cate 10 pana la 60,70,80,
    # apoi inca 20 pana la 100
    # verificand si acuratetea pe validare intre aceste perioade de antrenare
    NUM_EPOCHS = 30
    my_model.train(True)
    loss_sum = 0
    losses = 0
    for epoch in range(NUM_EPOCHS):
        print(f"--Epoch {epoch + 1}--")
        losses = 0
        loss_sum = 0
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = my_model(images)
            loss = cross_entropy_loss(out, labels)
            loss_sum += loss
            losses += 1
            optimizing_algo.zero_grad()
            loss.backward()
            optimizing_algo.step()

            if batch % 20 == 0:
                print(f"Batch index {batch}; learning rate {optimizing_algo.param_groups[0]['lr']}; loss: {loss.item():>8f}")

        reduce_lr.step(loss_sum / losses)
        print(f"mean loss: {loss_sum / losses}")

    # obtinerea predictiilor pentru setul de validare si acuratetea pe setul de validare
    correctly_predicted = 0
    test_loss = 0
    len_validation_set = len(val_loader.dataset)
    val_labels = []
    predicted_val = []
    my_model.to(DEVICE)
    my_model.eval()
    with torch.no_grad():
        for image_batch, labels_batch in val_loader:
            image_batch = image_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            pred = my_model(image_batch)
            test_loss += cross_entropy_loss(pred, labels_batch).item()
            correctly_predicted += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()
            predicted_val.extend(pred)
            val_labels.extend(labels_batch)

    correctly_predicted /= len_validation_set
    test_loss /= len_validation_set
    print(f"Accuracy: {(100 * correctly_predicted):>0.1f}%, Loss: {test_loss:>8f} \n")

    # incarcarea imaginilor de test
    test_data = DeepHallucinationDataSet(TEST_FOLDER, TEST_IMAGES_NAMES, mode="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

    # prezicerea claselor imaginilor din setul de testare
    my_model.to(DEVICE)
    my_model.eval()
    predictions_test = []
    with torch.no_grad():
        for image in test_loader:
            image = image.to(DEVICE)
            pred = my_model(image).argmax(1)
            predictions_test.extend(pred.tolist())

    # salvarea claselor prezise in fisierul csv care va fi submis
    # uneori am facut mai multe salvari intermediare ale predictiilor date de modelul meu
    write_results(images_names=os.listdir("/kaggle/input/unibuc-dhc-2023/test_images"),
                  pred_test=predictions_test,
                  file_name="ResNettry--17.06_-89.4 50 epochs")


