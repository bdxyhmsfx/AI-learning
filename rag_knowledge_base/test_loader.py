from rag_knowledge_base.loaders.document_loader import load_directory
documents = load_directory("data")
print("Document 总数量：", len(documents))
for document in documents:
    print("\n" + "=" * 60)
    print("source:",document["source"])
    print("file_type:",document["file_type"])
    print("page:",document["page"])
    print("title:",document["title"])
    print("\nTEXT:")
    print(document["text"][:300])
# from rag_knowledge_base.loaders.pdf_loader import load_pdf_file
# from rag_knowledge_base.loaders.text_loader import load_text_file
# from rag_knowledge_base.loaders.docx_loader import load_docx_file
#
# print("pdf-test")
# documents = load_pdf_file("data/attention.pdf")
# print("Document 数量：", len(documents))
# for document in documents:
#     print("\n" + "=" * 50)
#     print("source:",document["source"])
#     print("page:",document["page"])
#     print("\nTEXT:")
#     print(document["text"])
#
# print("\n\ntext,md-test")
# text_document = load_text_file("data/transformer.md")
# print(text_document["text"])
#
# print("\n\ndocx-test")
# docx_document = load_docx_file("data/rag.docx")
# print(docx_document["text"])