from src.information_retrieval_model import Kickstarter_Information_Retrieval_Model
from src.prediction_model import Kickstarter_Prediction_Model

from IPython.display import display
from ipywidgets import widgets
from IPython.display import clear_output

class kickstarter_input():
    def __init__(self, 
                 description = "e.g. 'I have a great idea...'", 
                 category = "e.g. 'game'", 
                 goal_USD = "e.g. '5000'",
                 days_to_deadline = "e.g. '21'"
                ):
        self.description = widgets.Text(description = 'Description',value = description)
        self.category = widgets.Dropdown(options = ['games','technology','film & video','art','fashion',
                                                    'music','publishing','design','comics','food',
                                                   'crafts','theater','photograph','journalism','dance'],
                                                    description='Category')
        self.goal_USD = widgets.Text(description = 'Amount',value = goal_USD)
        self.days_to_deadline = widgets.Text(description = 'Duration',value = days_to_deadline)
        
        self.description.on_submit(self.handle_submit)
        self.goal_USD.on_submit(self.handle_submit)
        self.days_to_deadline.on_submit(self.handle_submit)
        
        display(self.description, 
                self.category, 
                self.goal_USD, 
                self.days_to_deadline)
        
    def handle_submit(self, text):
        self.v = text.value
        return self.v


if __name__ == "__main__":
    submit = kickstarter_input()
    kir = Kickstarter_Information_Retrieval_Model()
    forecast = Kickstarter_Prediction_Model()
    
    button = widgets.Button(description="Submit")
    display(button)
    def on_button_clicked(b):
        clear_output()
        forecast.make_prediction(submit)
        kir.get_similar_project(submit)
    button.on_click(on_button_clicked)



