package com.nju.parser;

import com.github.javaparser.ParserConfiguration;
import com.nju.config.Config;

public class Main {
    public static void main(String[] args) {
        InfoParser parser = new InfoParser(new Config("E:\\Bundle\\Java源代码\\src\\java.base\\java", "output", ParserConfiguration.LanguageLevel.JAVA_11));
        parser.parse();
    }
}
