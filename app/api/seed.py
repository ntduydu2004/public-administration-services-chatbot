'''
seed.py users the models in model.py and populates the database with dummy content
'''

# ----------------
# Database imports
# ----------------
from helpers import (
    create_org_by_org_or_uuid,
    create_project_by_org,
    create_document_by_file_path
)
from config import (
    FILE_UPLOAD_PATH,
    logger
)
from util import (
    get_file_hash
)
import os

# --------------------
# Create organizations
# --------------------

organizations = [
    {
        'display_name': 'Vietnamese national public service portal',
        'namespace': 'vnpsp',
        'projects': [
            {
                'display_name': 'Birth registration',
                'docs': [
                    'birth_registration/birth_registration.md',
                    'birth_registration/birth_registration_instruction.md',
                    'birth_registration/birth_registration_QA.md',
                ]
            },
            {
                'display_name': 'Death registration',
                'docs': [
                    'death_registration/death_registration.md',
                    'death_registration/death_registration_instruction.md',
                    'death_registration/death_registration_QA.md',
                ]
            },
            {
                'display_name': 'Marriage registration',
                'docs': [
                    'marriage_registration/marriage_registration.md',
                    'marriage_registration/marriage_registration_instruction.md',
                    'marriage_registration/marriage_registration_QA.md',
                ]
            },
            {
                'display_name': 'Domestic adoption registration',
                'docs': [
                    'domestic_registration/domestic_adoption_registration.md',
                    'domestic_registration/domestic_adoption_registration_instruction.md',
                    'domestic_registration/domestic_adoption_registration_QA.md',
                ]  
            },
            {
                'display_name': 'Guardianship registration',
                'docs': [
                    'guardianship_registration/guardianship_registration.md',
                    'guardianship_registration/guardianship_registration_instruction.md',
                    'guardianship_registration/guardianship_registration_QA.md',
                ]
            },
            {
                'display_name': 'Permanent residence registration',
                'docs': [
                    'permanent_residence_registration/permanent_residence_registration.md',
                    'permanent_residence_registration/permanent_residence_registration_instruction.md',
                    'permanent_residence_registration/permanent_residence_registration_QA.md',
                ]
            },
            {
                'display_name': 'ID card issuance',
                'docs': [
                    'id_card_issuance/id_card_issuance.md',
                    'id_card_issuance/id_card_issuance_instruction.md',
                    'id_card_issuance/id_card_issuance_QA.md',
                ]
            },
            {
                'display_name': 'ID card re-issuance',
                'docs': [
                    'id_card_re-issuance/id_card_re-issuance.md',
                    'id_card_re-issuance/id_card_re-issuance_instruction.md',
                    'id_card_re-issuance/id_card_re-issuance_QA.md',
                ]
            },
            {
                'display_name': 'Vietnamese national public service portal (portal)',
                'docs': [
                    'public_services_portal/org-about_the_portal.md',
                    'public_services_portal/org-functions_of_the_portal.md',
                    'public_services_portal/org-usage_instructions.md',
                    'public_services_portal/org-agency_finding_instruction.md',
                ]
            }
        ]
    }
]

training_data_path = os.path.join(os.path.dirname(__file__), f'{FILE_UPLOAD_PATH}/training_data')

for org in organizations:

    org_obj = create_org_by_org_or_uuid(
        display_name=org['display_name'],
        namespace=org['namespace']
    )
    logger.debug(f'🏠  Created organization: {org_obj.display_name}')

    if 'projects' not in org:
        continue

    for project in org['projects']:
        project['organization'] = org_obj

        project_obj = create_project_by_org(
            organization_id=org_obj,
            display_name=project['display_name']
        )
        logger.debug(f'🗂️  Created project: {project_obj.display_name}')

        project_uuid = str(project_obj.uuid)
        org_uuid = str(org_obj.uuid)

        # if the directory does not exist, create it
        if not os.path.exists(os.path.join(FILE_UPLOAD_PATH, org_uuid, project_uuid)):
            os.mkdir(os.path.join(FILE_UPLOAD_PATH, org_uuid, project_uuid))

        if 'docs' not in project:
            continue

        for doc in project['docs']:
            file_path = os.path.join(training_data_path, doc)

            # check if file exists
            if os.path.isfile(file_path):
                file_hash = get_file_hash(file_path)
                create_document_by_file_path(
                    organization=org_obj,
                    project=project_obj,
                    file_path=file_path,
                    file_hash=file_hash
                )
                logger.info(f'  ✅  Created document: {doc}')
            else:
                logger.error(f' ❌  Document not found: {doc}')