"""Authentication forms with WTForms validation."""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import (DataRequired, Email, EqualTo, Length,
                                  ValidationError, Regexp)
from webapp.app.models import User

# Disable DNS deliverability check — only validates email format.
# email-validator 2.x performs DNS lookups by default which rejects
# non-public domains (e.g. .local, .test) used in development.
_email_validator = Email(check_deliverability=False)


class LoginForm(FlaskForm):
    email    = StringField("Email",    validators=[DataRequired(), _email_validator])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Keep me signed in")
    submit   = SubmitField("Sign In")


class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[
        DataRequired(),
        Length(3, 32),
        Regexp(r"^[A-Za-z0-9_]+$",
               message="Username may only contain letters, numbers and underscores."),
    ])
    email    = StringField("Email", validators=[
        DataRequired(),
        Email(check_deliverability=False),
    ])
    password = PasswordField("Password", validators=[
        DataRequired(),
        Length(8, 128, message="Password must be at least 8 characters."),
    ])
    confirm  = PasswordField("Confirm Password", validators=[
        DataRequired(),
        EqualTo("password", message="Passwords must match."),
    ])
    submit   = SubmitField("Create Account")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.lower()).first():
            raise ValidationError("That email is already registered.")

    def validate_username(self, field):
        if User.query.filter_by(username=field.data).first():
            raise ValidationError("Username already taken.")
