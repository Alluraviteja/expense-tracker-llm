from flask import Blueprint, request, jsonify
from .parser import parse_expense

bp = Blueprint('api', __name__)


@bp.route('/parse', methods=['POST'])
def parse():
    data = request.get_json()
    text = data.get("text", "")
    result = parse_expense(text)
    return jsonify(result)
