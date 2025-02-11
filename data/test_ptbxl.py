import unittest
import torch
from torch.utils.data import DataLoader

# Import the PTBXLDataModule class
from ptbxl import PTBXLDataModule  # Replace `your_module` with the actual module name
from ptbxl import PTBXL  # Assuming PTBXL is also in the same module

class TestPTBXLDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the PTBXLDataModule instance before running tests."""
        cls.root_dir = "/global/D1/homes/vajira/data/ecg/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"  # Update this with actual path

        cls.dm = PTBXLDataModule(
            root_dir=cls.root_dir,
            train_folds_cl=[1, 2],   # Folds for contrastive learning
            train_folds_gt=[3, 4, 5],  # Folds for ground task training
            val_folds_gt=[6, 7],     # Folds for validation
            test_folds_gt=[8, 9, 10], # Folds for testing
            sampling_rate=100,
            batch_size=16
        )

    def test_contrastive_learning_loader(self):
        """Test dataloader for contrastive learning."""
        self.dm.setup(stage="contrastive")
        loader = self.dm.train_dataloader(task="contrastive")
        self.assertIsInstance(loader, DataLoader)

        batch = next(iter(loader))
        self.assertIn("ecg", batch)
        self.assertTrue(isinstance(batch["ecg"], torch.Tensor))
        print("✅ Contrastive Learning Loader Test Passed")

    def test_multilabel_classification_loaders(self):
        """Test dataloaders for multilabel classification."""
        self.dm.setup(stage="fit")

        # Train dataloader
        train_loader = self.dm.train_dataloader(task="ground")
        self.assertIsInstance(train_loader, DataLoader)

        batch = next(iter(train_loader))
        self.assertIn("ecg", batch)
        self.assertIn("class", batch)
        self.assertTrue(isinstance(batch["ecg"], torch.Tensor))
        self.assertTrue(isinstance(batch["class"], torch.Tensor))
        print("✅ Train Loader Test Passed")

        # Validation dataloader
        val_loader = self.dm.val_dataloader()
        self.assertIsInstance(val_loader, DataLoader)
        print("✅ Validation Loader Test Passed")

    def test_test_loader(self):
        """Test dataloader for the test set."""
        self.dm.setup(stage="test")
        test_loader = self.dm.test_dataloader()
        self.assertIsInstance(test_loader, DataLoader)

        batch = next(iter(test_loader))
        self.assertIn("ecg", batch)
        self.assertIn("class", batch)
        self.assertTrue(isinstance(batch["ecg"], torch.Tensor))
        self.assertTrue(isinstance(batch["class"], torch.Tensor))
        print("✅ Test Loader Test Passed")

if __name__ == "__main__":
    unittest.main()