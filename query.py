from vncorenlp import VnCoreNLP
import json
annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)

loc_query = "tên địa danh bao gồm các hành tinh, quốc gia, vùng lãnh thổ, châu lục, làng, thị trấn, thành phố, tỉnh, giáo khu, giáo xứ, núi, rừng, sông, suối, hồ, biển, vịnh, vũng, eo biển, đại dương, cầu, đường, cảng, đập, lâu đài, tháp, phòng trưng bày, hội trường, viện dưỡng lão, địa điểm, địa chỉ thương mại"
per_query = "tên người bao gồm tên, tên đệm và họ của một người, tên động vật và các nhân vật hư cấu, bí danh, biệt danh"
org_query = "tên tổ chức bao gồm cơ quan bộ ngành, uỷ ban nhân dân, hội đồng nhân dân, toà án, cơ quan báo chí, hội nghề nghiệp, đoàn thể chính trị, phòng ban, công ty,  tổ chức chính trị, tạp chí, báo, trường học, tổ chức từ thiện, câu lạc bộ thể thao, nhà hát"
misc_query = "tên gọi khác bao gồm quốc tịch, ngôn ngữ, môn học, danh hiệu, cuộc thi"

queries = {
    'LOC': ' '.join(sum(annotator.tokenize(loc_query), [])),
    'PER': ' '.join(sum(annotator.tokenize(per_query), [])),
    'ORG': ' '.join(sum(annotator.tokenize(org_query), [])),
    'MISC': ' '.join(sum(annotator.tokenize(misc_query), [])),
}

with open('queries.json', mode='w', encoding='utf8') as f:
    json.dump(queries, f, indent=4, ensure_ascii=False)

