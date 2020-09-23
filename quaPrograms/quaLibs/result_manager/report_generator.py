import os

import dominate
from dominate import tags as dom_tags


def make_report(result_folder_list):
    pass


def make_result_report(result_folder_list):
    doc = dominate.document(title='QM results report')

    with doc.head:
        dom_tags.link(rel='stylesheet', href='style.css')
        # dom_tags.script(type='text/javascript', src='script.js')

    with doc:
        with dom_tags.div(id='header').add(dom_tags.ol()):
            dom_tags.h1('QM Results report')

        with dom_tags.div():
            dom_tags.attr(cls='body')

            for folder in result_folder_list:
                with dom_tags.div(id=folder):
                    dom_tags.p(f"{folder}")
                    for file_name in os.listdir(folder):
                        with dom_tags.div(id=file_name):
                            if file_name.split('.')[1] == 'npz':
                                dom_tags.p(file_name)
                            else:
                                with open(os.path.join(folder, file_name), 'r') as f:
                                    dom_tags.p(f.read())

    with open('report.html', 'w') as report_html:
        report_html.write(doc.__str__())


def get_results_in_path(path):
    folder_list = [x[0] for x in os.walk(path)]
    res_folder_list = []
    for folder in folder_list:
        if os.path.exists(os.path.join(folder, 'result.log')):
            res_folder_list.append(folder)
            print('results folder found!')
    return res_folder_list
