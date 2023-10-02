"Definition of major forms used in flask"
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired

class RaiseQuestionForm(FlaskForm):
    """
    Combination of needed flask fields 

    Parameters
    ----------
    FlaskForm : flask form
        Import of Flaskform class
    """
    question = StringField("question",render_kw={'style': "width: 50ch; height: 5ch"}, validators = [DataRequired()])
    submit = SubmitField("Submit question")
    new = BooleanField("Continue conversation")