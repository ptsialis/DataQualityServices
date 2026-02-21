import logging
import zipfile
import io
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnlabelledDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),         # Standard input size for most models
            transforms.ToTensor(),                 # Converts to torch.Tensor and scales to [0,1]
        ])
        
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            # Infer the single image extension type present
            all_files = archive.namelist()
            image_ext = next((ext for ext in ['.jpg', '.jpeg', '.png']
                            if any(f.lower().endswith(ext) for f in all_files)), None)
            if image_ext is None:
                raise RuntimeError("No .jpg/.jpeg/.png files found in ZIP.")
            
            self.image_paths = sorted([
                f for f in all_files if f.lower().endswith(image_ext)
            ])
        
        self.archive = zipfile.ZipFile(self.zip_path, 'r')  # keep open for __getitem__

    def __len__(self):
        return len(self.image_paths)

    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]
    #     with self.archive.open(img_path) as file:
    #         img = Image.open(io.BytesIO(file.read())).convert("RGB")
    #     if self.transform:
    #         img = self.transform(img)
    #     else:
    #         raise RuntimeError("No transform provided for UnlabelledDataset.")
    #     return img # , idx
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            with self.archive.open(img_path) as file:
                img_bytes = file.read()
                if not img_bytes:
                    raise ValueError(f"Empty file: {img_path}")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except (UnidentifiedImageError, ValueError, OSError) as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            # Option 1: Skip to the next image (cyclic fallback)
            next_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(next_idx)

            # Option 2: Raise exception instead of skipping
            # raise RuntimeError(f"Image at {img_path} could not be loaded: {e}")

        if self.transform:
            img = self.transform(img)
        else:
            raise RuntimeError("No transform provided for UnlabelledDataset.")

        return img  # , idx

