import time
from flask_restx import Namespace, Resource
from flask import Flask, request, jsonify
from skops.io import load, get_untrusted_types
from huggingface_hub import hf_hub_download
from pathlib import Path
import spacy
import glob
import os
from flask_restx import Namespace, Resource, reqparse

import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('PredictCpv')

api = Namespace('Cpv operations')

# ======================================================
# Define parsers to take inputs from user
# ======================================================
cpv_parser = reqparse.RequestParser()
cpv_parser.add_argument("text_to_infer",
                        help="Text to whose CPV code is to be inferred",
                        required=True)

############################################################################
# @TODO: Move this to ../core/cpv_classifier.py. Create a class CpvClassifier that receives the model and vectorizer as parameters in the constructor such that the loading are only loaded once (in the constructor). The function "load_model_from_hf" should not be an endpoint.
def load_model_from_hf():
    try:
        # Base path where the model and vectorizer are expected to be found
        base_path = "/models/models--erick4556--cpv-3digits/snapshots/" # @ TODO: this should be given as a parameter

        # Search for .skops files within the directory structure
        model_files = glob.glob(base_path + '**/model.skops', recursive=True)
        vectorizer_files = glob.glob(
            base_path + '**/vectorizer.skops', recursive=True)

        if not model_files or not vectorizer_files:
            logger.error(f"-- -- Model files not found in {base_path}")
            return None, None
            
        logger.info(f"-- -- Model files found: {model_files}")
        logger.info(f"-- -- Vectorizer files found: {vectorizer_files}")

        # Assuming the first found files are the ones needed
        # @TODO: This should not be assumed, some sort of versioning should be implemented
        model_path = model_files[0]
        vectorizer_path = vectorizer_files[0]

        # Get untrusted types if necessary
        untrusted_types_model = get_untrusted_types(file=model_path)
        untrusted_types_vectorizer = get_untrusted_types(
            file=vectorizer_path)
        logger.info(f"-- -- Untrusted types for model: {untrusted_types_model}")
        logger.info(f"-- -- Untrusted types for vectorizer: {untrusted_types_vectorizer}")

        # Load the model and vectorizer
        model = load(model_path, trusted=untrusted_types_model)
        vectorizer = load(
            vectorizer_path, trusted=untrusted_types_vectorizer)
        logger.info("-- -- Model and vectorizer loaded successfully")

        return model, vectorizer

    except Exception as e:
        print(f"Error loading models from local storage: {e}")
        return None, None

def preprocess_text(text):
    # Procesamiento del texto para llevarlo a su forma base (lematización)
    doc = nlp(text.lower())
    words = [
        token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(words)

def predict_description(description):

    # Preprocesar descripción y vectorizar
    processed_description = preprocess_text(description)
    description_vectorized = vectorizer.transform([processed_description])

    # Realizar predicción
    probs = model.predict_proba(description_vectorized)
    print(probs)

    max_prob = 0
    predicted_code_index = None

    for i, model_probs in enumerate(probs):
        # Index 1 para la clase positiva
        current_max_prob = model_probs[0][1]
        if current_max_prob > max_prob:
            max_prob = current_max_prob
            predicted_code_index = i

    predicted_code = cpv_codes[predicted_code_index]
    result = {
        "cpv_predicted": predicted_code,
        "probability": round(max_prob, 2)
    }
    print(f"Result: {result}")
    return result

# @TODO: Move this to a configuration file or to the constructor of the class CpvClassifier
cpv_codes = [
    '030', '031', '032', '033', '034', '090', '091', '092', '093', '140', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '163', '164', '165', '166', '167', '168', '180', '181', '182', '183', '184', '185', '186', '188', '189', '190', '191', '192', '194', '195', '196', '197', '220', '221', '222', '223', '224', '225', '226', '228', '229', '240', '241', '242', '243', '244', '245', '246', '249', '300', '301', '302', '310', '311', '312', '313', '314', '315', '316', '317', '320', '322', '323', '324', '325', '330', '331', '336', '337', '339', '340', '341', '342', '343', '344', '345', '346', '347', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '370', '373', '374', '375', '378', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '395', '397', '398', '410', '411', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '440', '441', '442', '443', '444', '445', '446', '448', '449',
    '450', '451', '452', '453', '454', '455', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '500', '501', '502', '503', '504', '505', '506', '507', '508', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '550', '551', '552', '553', '554', '555', '559', '600', '601', '602', '603', '604', '605', '606', '630', '631', '635', '637', '640', '641', '642', '650', '651', '652', '653', '654', '655', '660', '661', '665', '666', '667', '700', '701', '702', '703', '710', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '750', '751', '752', '753', '760', '761', '762', '763', '764', '765', '766', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', '800', '801', '802', '803', '804', '805', '806', '850', '851', '852', '853', '900', '904', '905', '906', '907', '909', '920', '921', '922', '923', '924', '925', '926', '927', '980', '981', '982', '983', '985', '989']

# @TODO: Move this to the constructor of the class CpvClassifier as attribute
nlp = spacy.load('es_dep_news_trf')
############################################################################

# @TODO: model and vectorizer should be attributes of the class CpvClassifier
# Here an instance of the class CpvClassifier should be created and the method predict_description should be called
# ======================================================
# Create CpvClassifier object
# ======================================================
model, vectorizer = load_model_from_hf()


@api.route('/predictCpv')
class predictCpv(Resource):
    @api.doc(
        parser=cpv_parser,
        responses={
            200: 'Success: CPV code generated successfully',
            501: 'CPV error: ....', #@TODO: Add a proper error message
        }
    )
    def post(self):
        
        start_time = time.time()
        
        args = cpv_parser.parse_args()
        
        try:
            result = predict_description(args['text_to_infer'])
            
            end_time = time.time() - start_time
            
            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time,
            }
            response = {
                "responseHeader": responseHeader,
                "response": result
            }
            logger.info(
                f"-- -- CPV code generated successfully:{result}")
            
            return response, sc
        
        except Exception as e:
            end_time = time.time() - start_time
            sc = 502
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": str(e)
            }
            response = {
                "responseHeader": responseHeader
            }
            logger.error(
                f"-- -- CPV code generation error: {str(e)}")
            
            return response, sc