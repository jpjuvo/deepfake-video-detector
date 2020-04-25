import os
import json

class RecurrentModelConfig:
    
    def __init__(self,
                 len_sequence=4,
                 min_frame_index=10,
                 max_frame_index=16):
        assert len_sequence is not None, 'Len sequence is None'
        assert len_sequence > 1, 'Len sequuence is smaller than 2'

        self.config_dict = {'len_sequence':len_sequence,
                            'min_frame_index':min_frame_index,
                            'max_frame_index':max_frame_index}
        print("RecurrrentModelConfig: len_sequence {0}, min_frame_index {1}, max_frame_index {2}".format(self.getLenSequence(),
                                                                                                         self.getMinFrameIndex(),
                                                                                                         self.getMaxFrameIndex()))

    @classmethod
    def fromDir(cls, dir_path):
        config_path = os.path.join(dir_path,'config.json')
        len_sequence, min_frame_index, max_frame_index = None, None, None
        with open(config_path) as json_file:
            data = json.load(json_file)
            len_sequence = data['len_sequence']
            min_frame_index = data['min_frame_index']
            max_frame_index = data['max_frame_index']
        return cls(len_sequence, min_frame_index, max_frame_index)

    def toDir(self, dir_path):
        config_path = os.path.join(dir_path,'config.json')
        with open(config_path, 'w') as outfile:
            json.dump(self.config_dict, outfile)

    def getLenSequence(self):
        return self.config_dict['len_sequence']
    
    def getMinFrameIndex(self):
        return self.config_dict['min_frame_index']

    def getMaxFrameIndex(self):
        return self.config_dict['max_frame_index']