import streamlit as st
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image

from models import dcgan_G, dcgan_D, cgan_G, cgan_D

device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
img_size = 64

species = [
    "GREEN CELLED CATTLEHEART",
    "AFRICAN GIANT SWALLOWTAIL"
]

transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

st.title("Butterfly GAN Generator")


# MODEL SELECTION

model_type = st.selectbox(
    "Select GAN Model",
    ["DCGAN","CGAN"]
)


# LOAD MODELS

if model_type == "DCGAN":

    G = dcgan_G().to(device)
    D = dcgan_D().to(device)

    G.load_state_dict(torch.load("weights/dcgan_G.pth",map_location=device))
    D.load_state_dict(torch.load("weights/dcgan_D.pth",map_location=device))


else:

    G = cgan_G().to(device)
    D = cgan_D().to(device)

    G.load_state_dict(torch.load("weights/cgan_G.pth",map_location=device))
    D.load_state_dict(torch.load("weights/cgan_D.pth",map_location=device))


G.eval()
D.eval()


# CLASS SELECT (ONLY CGAN)

if model_type == "CGAN":

    class_id = st.selectbox(
        "Select Butterfly Species",
        range(len(species)),
        format_func=lambda x: species[x]
    )

else:
    class_id = None


num_images = st.slider("Number of images",1,16,4)


# GENERATE IMAGES

if st.button("Generate"):

    if model_type == "DCGAN":

        noise = torch.randn(num_images,latent_dim,1,1).to(device)

        with torch.no_grad():
            imgs = G(noise)

    else:

        noise = torch.randn(num_images,latent_dim).to(device)
        labels = torch.tensor([class_id]*num_images).to(device)

        with torch.no_grad():
            imgs = G(noise,labels)

    grid = make_grid(imgs,nrow=4,normalize=True)

    st.image(grid.permute(1,2,0).cpu().numpy(), width=700)


# REAL / FAKE CLASSIFIER

st.subheader("Upload Butterfly Image")

uploaded_file = st.file_uploader(
    "Upload image to check REAL or FAKE",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        if model_type == "DCGAN":
            pred = D(img_tensor)

        else:
            label = torch.tensor([class_id]).to(device)
            pred = D(img_tensor,label)

    score = pred.item()

    if score > 0.5:
        st.success("REAL IMAGE")
    else:
        st.error("FAKE IMAGE")

    st.write("Confidence:",round(score,3))

# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from PIL import Image

# from models import (
#     dcgan_Generator, dcgan_Discriminator,
#     cgan_Generator, cgan_Discriminator,
#     vanilla_Generator, vanilla_Discriminator
# )

# device = "cpu"

# latent_dim = 100
# img_size = 64
# num_classes = 10

# transform = transforms.Compose([
#     transforms.Resize((img_size,img_size)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3,[0.5]*3)
# ])

# st.title("Butterfly Generator using GAN")

# # Model Selection

# model_option = st.selectbox(
#     "Choose GAN Model",
#     ["DCGAN", "CGAN", "VGAN"]
# )

# if model_option == "DCGAN":

#     G = dcgan_Generator().to(device)
#     D = dcgan_Discriminator().to(device)

#     G.load_state_dict(torch.load("weights/dcgan_generator.pth", map_location=device))
#     D.load_state_dict(torch.load("weights/dcgan_discriminator.pth", map_location=device))

# elif model_option == "CGAN":

#     G = cgan_Generator().to(device)
#     D = cgan_Discriminator().to(device)

#     G.load_state_dict(torch.load("weights/cgan_generator.pth", map_location=device))
#     D.load_state_dict(torch.load("weights/cgan_discriminator.pth", map_location=device))

# elif model_option == "VGAN":

#     G = vanilla_Generator().to(device)
#     D = vanilla_Discriminator().to(device)

#     G.load_state_dict(torch.load("weights/vgan_generator.pth", map_location=device))
#     D.load_state_dict(torch.load("weights/vgan_discriminator.pth", map_location=device))

# G.eval()
# D.eval()

# # Generate Images

# st.subheader("Generate Images")

# if st.button("Generate Images"):

#     noise = torch.randn(64,latent_dim,1,1).to(device)

#     with torch.no_grad():

#         if model_option == "DCGAN":
#             fake = G(noise)

#         elif model_option == "VGAN":
#             fake = G(noise.view(64, latent_dim))
#             fake = fake.view(64,3,img_size,img_size)

#         else:
#             labels = torch.randint(0,num_classes,(64,)).to(device)
#             fake = G(noise.view(64,latent_dim),labels)

#     grid = make_grid(fake,nrow=8,normalize=True)

#     fig,ax = plt.subplots()
#     ax.imshow(grid.permute(1,2,0).cpu())
#     ax.axis("off")

#     st.pyplot(fig)

# # Upload Image

# st.subheader("Upload Image to Classify")

# uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

# if uploaded_file:

#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image")

#     img_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():

#         if model_option == "DCGAN":
#             pred = D(img_tensor)

#         elif model_option == "VGAN":
#             pred = D(img_tensor.view(1,-1))

#         else:
#             label = torch.randint(0,num_classes,(1,)).to(device)
#             pred = D(img_tensor,label)

#     score = pred.item()

#     if score > 0.5:
#         st.success("Prediction: REAL IMAGE")
#     else:
#         st.error("Prediction: FAKE IMAGE")

#     st.write("Confidence:", round(score,3))


# # Compute Dynamic Accuracy

# def compute_accuracy(G, D):

#     correct = 0
#     total = 0

#     # Generate fake images
#     noise = torch.randn(32, latent_dim, 1, 1).to(device)

#     with torch.no_grad():

#         if model_option == "DCGAN":
#             fake = G(noise)
#             fake_pred = D(fake)
        
#         elif model_option == "VGAN":
#             fake = G(noise.view(32,latent_dim))
#             fake = fake.view(32,3,img_size,img_size)
#             fake_pred = D(fake.view(32,-1))

#         else:
#             labels = torch.randint(0,num_classes,(32,)).to(device)
#             fake = G(noise.view(32,latent_dim), labels)
#             fake_pred = D(fake, labels)

#     fake_labels = torch.zeros_like(fake_pred)

#     correct += (fake_pred < 0.5).sum().item()
#     total += fake_pred.size(0)

#     accuracy = correct / total

#     return accuracy

# acc = compute_accuracy(G,D)

# st.metric("Discriminator Accuracy", f"{acc*100:.2f}%")