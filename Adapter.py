from pathlib import Path

import cv2
import numpy as np
from loguru import logger


from Final2x_core.src.SRFactory import SRFactory
from Final2x_core.src.utils.getConfig import SRCONFIG
from Final2x_core.src.utils.progressLog import PrintProgressLog

projectPATH = Path(__file__).resolve().parent.absolute()

class Adapter():
    def __init__(self, config_json):
        self.config_json = config_json
        self.config = SRCONFIG()
        self.config.gpuid = self.config_json["gpuid"]
        self.config.tta = self.config_json["tta"]
        self.config.model = self.config_json["model"]
        self.config.modelscale = self.config_json["modelscale"]
        self.config.modelnoise = self.config_json["modelnoise"]
        self.config.inputpath = ['dummy', 'input']
        # targetscale should be set after modelscale
        self.config.targetscale = self.config_json["targetscale"]

        self.sr = SRFactory.getSR()
    
    def queue(self, src_path, dest_path):
        try:
            img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception('Failed to decode image.')
        except Exception as e:
            logger.error(str(e))
            logger.warning("CV2 load image failed: " + src_path + ", skip. ")
            logger.warning("______Skip_Image______: " + src_path)
            PrintProgressLog().skipProgress()
        logger.info("Processing: " + src_path + ", save to: " + dest_path)
        img = self.sr.process(img)
        dest_ext = Path(dest_path).suffix
        cv2.imencode(dest_ext, img)[1].tofile(dest_path)
        logger.success("______Process_Completed______: " + src_path)

if __name__ == '__main__':
    json_data = {
        'gpuid': 0,
        'model': 'RealESRGAN-anime',
        'modelscale': 2,
        'modelnoise': 0,
        'targetscale': 4,
        'tta': False
    }
    adapter = Adapter(json_data)
    adapter.queue('test.png', 'test2.png')