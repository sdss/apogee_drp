import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


def send(send_to, subject='', message='', msgtype='html', files=None):
    """
    Send an email message.

    Parameters
    ----------
    send_to : str or list
         List of email addresses.
    subject : str, optional
         Subject line.
    message : str, optional
         The message text.
    msgtype : str, optional
         The message text type (html or plan). html by default.
    files : list of str, optional
         List of files to attach.

    Returns
    -------
    An email is send to the address.

    Examples
    --------
    send('somebody@somewhere.com','This is a test','Important message')

    """

    if isinstance(send_to, list)==False:
        send_to = [send_to]

    hostname = 'sdss.org'
    send_from = 'noreply.dailyapogee@%s' % hostname

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message,msgtype))

    if files is not None:
        if isinstance(files,list) is False: files=[files]  # make sure it's a list
        for f in files or []:
            with open(f, "rb") as fil:
                part = MIMEApplication(
                    fil.read(),
                    Name=basename(f)
                )
            # After the file is closed
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)


    smtp = smtplib.SMTP('localhost')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
