"""
Index
"""

import logging
from flask import Blueprint, render_template

bp = Blueprint("bp", __name__, template_folder="../templates")

log = logging.getLogger(__name__)  # noqa: E402

logging.basicConfig(level=logging.INFO)


@bp.route("/")
def show():
    """
    Blueprint for index page
    :return:
    """
    log.info("accessing index")
    # read if value in csv fi
    return render_template("index.html")
