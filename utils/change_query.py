DEFAULT_QUERY_MAPPING = {
    'PER': "thực_thể người bao_gồm tên , tên đệm và họ của người , tên động_vật , nhân_vật hư_cấu , bí_danh",
    'LOC': "thực_thể địa_danh bao_gồm tên gọi các hành_tinh , thực_thể tự_nhiên , địa_lí lịch_sử , vùng quần_cư , công_trình kiến_trúc xây dụng , địa_điểm , địa_chỉ",
    'ORG': "thực_thể tổ_chức bao_gồm các cơ_quan chính_phủ , công_ty , thương_hiệu , tổ_chức chính_trị , ấn_phẩm , tổ_chức công_cộng",
    'MISC': "thực_thể bao_gồm quốc_tịch , ngôn_ngữ , môn_học , danh_hiệu , cuộc thi",
}


def change_query(data: list, query_mapping: dict = None):
    if query_mapping is None:
        return query_mapping

    new_data = []
    for item in data:
        new_item = {
            **item,
            'query': query_mapping.get(item.get('entity_label'), item.get('query'))
        }
        new_data.append(new_item)
    return new_data
