class CrossEncoderTripletCollator:
    """
    Process triplets for the CrossEncoder
    Attributes
    ----------
    tokenizer: tokenizers.Tokenizer
        a pretrained Huggingface's tokenizer
    max_length:
        maximum lenght of each query, document pair joined together
    """
    def __init__(self, tokenizer, max_length):
        """
        Constructing the CrossEncoderTripletCollator
        Parameters
        ----------
        tokenizer: tokenizers.Tokenzier 
        max_length: int 
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Tokenize a list of (query, positive document, negative document) triplets in a batch
        Parameters
        ----------
        batch: list[(text, text, text)]
            a list of text triplets
        Returns
        -------
        a dictionary formated as {"pos_pairs": pos_pairs, "neg_pairs": neg_pairs} where:
        - pos_pairs: pairs of (query, positive document) jointly tokenzied by the tokenizer
        - neg_pairs: pairs of (query, negative document) jointly tokenizerd by the tokenizer 
        """
        pos_pairs = []
        neg_pairs = []
        for query, pos, neg in batch:
            pos_pairs.append((query, pos))
            neg_pairs.append((query, neg))
        pos_pairs = self.tokenizer(
            pos_pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg_pairs = self.tokenizer(
            neg_pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"pos_pairs": pos_pairs, "neg_pairs": neg_pairs}


class BiEncoderTripletCollator:
    """
    Process triplets for (Dense/Sparse)BiEncoder
    Attributes
    ----------
    tokenizer: tokenizers.Tokenizer
        a pretrained HuggingFace's tokenizer
    query_max_length: int 
        maximum length of a query. Token beyond this limit will be truncated by the tokenizer
    doc_max_length: int 
        maximum length of a document. Token beyond this limit will be truncated by the tokenizer
    """
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        """
        Tokenize a list of (query, positive document, negative document) triplets in a batch
        Parameters
        ----------
        batch: list[(text, text, text)]
            a list of text triplets
        Returns
        -------
        a dictionary formated as {"queries": queries, "pos_docs": pos_docs, "neg_docs": neg_docs} where:
        - queries: all queries in the input batch tokenized by the tokenizer
        - pos_docs: all documents in the input batch tokenized by the tokenizer
        - neg_docs: all documents in the input batch tokenized by the tokenizer
        """
        queries = []
        pos_docs = []
        neg_docs = []
        for query, pos, neg in batch:
            queries.append(query)
            pos_docs.append(pos)
            neg_docs.append(neg)
        queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        pos_docs = self.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        neg_docs = self.tokenizer(
            neg_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        return {"queries": queries, "pos_docs": pos_docs, "neg_docs": neg_docs}

class DistillModelTripletCollator:
    """
    Process triplets for DistillModel
    Attributes
    ----------
    student_tokenizer: tokenizers.Tokenizer
        a pretrained Huggingface's tokenizer
    teacher_tokenizer: tokenizers.Tokenizer
        a pretrained Huggingface's tokenizer
    max_length:
        maximum lenght of each query, document pair joined together
    """
    def __init__(self, student_tokenizer, teacher_tokenizer, query_max_length, doc_max_length):
        """
        Constructing the DistillModelTripletCollator
        Parameters
        ----------
        tokenizer: tokenizers.Tokenzier 
        max_length: int 
        """
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        """
        Tokenize a list of (query, positive document, negative document) triplets in a batch
        Parameters
        ----------
        batch: list[(text, text, text)]
            a list of text triplets
        Returns
        -------
        two dictionary formated as {"queries": queries, "pos_docs": pos_docs, "neg_docs": neg_docs} where:
        - queries: all queries in the input batch tokenized by the tokenizer
        - pos_docs: all documents in the input batch tokenized by the tokenizer
        - neg_docs: all documents in the input batch tokenized by the tokenizer
        """
        student_queries = []
        student_pos_docs = []
        student_neg_docs = []
        teacher_queries = []
        teacher_pos_docs = []
        teacher_neg_docs = []
        for query, pos, neg in batch:
            student_queries.append(query)
            student_pos_docs.append(pos)
            student_neg_docs.append(neg)
            teacher_queries.append(query)
            teacher_pos_docs.append(pos)
            teacher_neg_docs.append(neg)
        
        student_queries = self.student_tokenizer(
            student_queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        student_pos_docs = self.student_tokenizer(
            student_pos_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        student_neg_docs = self.student_tokenizer(
            student_neg_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        teacher_queries = self.teacher_tokenizer(
            teacher_queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        teacher_pos_docs = self.teacher_tokenizer(
            teacher_pos_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        teacher_neg_docs = self.teacher_tokenizer(
            teacher_neg_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        return {
            "student_queries": student_queries,
            "student_pos_docs": student_pos_docs,
            "student_neg_docs": student_neg_docs,
            "teacher_queries": teacher_queries,
            "teacher_pos_docs": teacher_pos_docs,
            "teacher_neg_docs": teacher_neg_docs,
        }
