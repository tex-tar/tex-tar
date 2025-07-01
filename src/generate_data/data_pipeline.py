import yaml 
from src.generate_data.generate_raw_json import DoctrProcessor
from src.generate_data.contextwindow import ContextCropGenerator
from src.generate_data.extract_cw import ContextCWExtractor
def main():
    #generate bboxes
    config_path = "config/data_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    processor = DoctrProcessor(cfg)
    processor.process_images()

    # make crops ; select the N nearest neighbours to the word
    extractor = ContextCropGenerator(cfg)
    extractor.run()

    #extract context windows
    cw = ContextCWExtractor(cfg)
    cw.run()

if __name__ == "__main__":
    main()