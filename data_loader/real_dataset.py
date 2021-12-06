import logging
import numpy as np
import pandas as pd

class RealDataset(object):
    """

    """
    _logger = logging.getLogger(__name__)

    def __init__(self):

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):

        cell_cd3cd28_aktinhib = pd.read_excel('data_loader/cell_cd3cd28_aktinhib.xls')
        cell_cd3cd28_u0126 = pd.read_excel('data_loader/cell_cd3cd28_u0126.xls')
       
        self.cell_dataset = []

        cell_sample_size = 30
        cell_num_subjects = 25
        for df in [cell_cd3cd28_u0126,cell_cd3cd28_aktinhib]:
            cell_data = []  
            
            start = 0
        
            for _ in range(cell_num_subjects):
                cell_subject = df.iloc[start:start + cell_sample_size, :]
                data = cell_subject.to_numpy()
                cell_data.append(data)
                
                start += cell_sample_size

            self.cell_dataset.append(np.array(cell_data))

        
if __name__ == "__main__":
    data = RealDataset()


        
        
