import yaml 
from src.generate_data.generate_raw_json import DoctrProcessor
from src.generate_data.contextwindow import ContextCropGenerator
from src.generate_data.extract_cw import ContextCWExtractor
from src.generate_data.organise import organize_images_by_number

def main():
    #generate bboxes
    config_path = "config/data_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if not cfg['skip_bounding_box_detection']:
        processor = DoctrProcessor(cfg)
        processor.process_images()

    # make crops ; select the N nearest neighbours to the word
    extractor = ContextCropGenerator(cfg)
    extractor.run()

    #extract context windows
    cw = ContextCWExtractor(cfg)
    cw.run()
    
    organize_images_by_number(source_folder=cfg['organize']['source_folder'],dest_folder=cfg['organize']['dest_folder'])

if __name__ == "__main__":
    main()