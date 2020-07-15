package com.nju.config;

import com.github.javaparser.ParserConfiguration;

public class Config {
    private String inputPath;   // 指定输入java文件目录
    private String outputPath;  // 指定输出文件目录

    private ParserConfiguration.LanguageLevel languageLevel;  // 解析的目标文件使用的Java版本

    public Config(String inputPath, String outputPath,
                  ParserConfiguration.LanguageLevel languageLevel) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.languageLevel = languageLevel;
    }

    public String getInputPath() {
        return inputPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public ParserConfiguration.LanguageLevel getLanguageLevel() {
        return languageLevel;
    }
}
