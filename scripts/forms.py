"Definition of major forms used in flask"
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
# from wtforms.widgets import TextArea
from wtforms.validators import DataRequired

class RaiseQuestionForm(FlaskForm):
    """
    Combination of needed flask fields 

    Parameters
    ----------
    FlaskForm : flask form
        Import of Flaskform class
    """
    question = StringField(label="question",
                           render_kw={"style": "width: 100%; height: 100%;"},
                           validators = [DataRequired()])
    submit = SubmitField(label="Continue old chat",
                         render_kw={"style": "width: 100%; text-align: justify;"}
                         )
    new_conv = SubmitField(label="Start new chat",
                           render_kw={"style": "width: 100%; text-align: justify;"},
                           description = "This is a test")
    new = BooleanField("Continue conversation")
