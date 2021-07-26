import smtplib
import requests

def send_mail(receiver, subject, content):
  gmail_user = 'PythonMailLogger@gmail.com'  
  gmail_password = 'RMzdm^_#3X54^N5Q'

  sent_from = gmail_user  
  to = [receiver]  
  subject = subject
  body = content


  email_text = "\r\n".join([
    "From: %s",
    "To: %s",
    "Subject: %s",
    "",
    "%s"
    ]) % (sent_from, ", ".join(to), subject, body)

  try:  
      server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
      server.ehlo()
      server.login(gmail_user, gmail_password)
      server.sendmail(sent_from, to, email_text)
      server.close()
  except Exception as e:  
      print('Something went wrong...')
      try:
          requests.get('https://www.google.com', timeout=5)
          print('Has connection')
      except requests.ConnectionError:
          print('No connection')
      raise e
