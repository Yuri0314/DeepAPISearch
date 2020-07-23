package com.nju.parser;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.javadoc.Javadoc;
import com.github.javaparser.javadoc.description.JavadocDescription;
import com.nju.config.Config;
import com.nju.util.ParserUtil;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class InfoParser {
    private String parseFolder; // 需要解析的java文件所在目录
    private String resDataPath; // 结果数据所在路径

    public InfoParser(Config config) {
        this.parseFolder = config.getInputPath();
        this.resDataPath = config.getOutputPath();
        new File(resDataPath).mkdir();
        StaticJavaParser.getConfiguration().setLanguageLevel(config.getLanguageLevel());
    }

    public void parse() {
            parseIteratable(new File(this.parseFolder));
    }

    private void parseIteratable(File folder){
        if (!folder.exists()) {
            System.err.println("输入文件夹" + folder.getAbsolutePath() + "不存在！");
            return;
        }
        for (File file : folder.listFiles()) {
            if (file.isDirectory())
                parseIteratable(file);
            else if (file.getName().endsWith(".java"))
                parseFile(file);
        }
    }

    public void parseFile(File file){
        CompilationUnit cu = null;
        List<String> infoList;
        try {
            cu = StaticJavaParser.parse(file);
            infoList = new ArrayList<>();
            Optional<PackageDeclaration> op = cu.getPackageDeclaration();
            if (!op.isPresent()) return;
            String packageName = op.get().getNameAsString();
            for (TypeDeclaration type : cu.getTypes()) {
                if (type.isClassOrInterfaceDeclaration())
                    parseClassOrInterface(type.getNameAsString(),
                            type, infoList);
            }
            saveToFile(infoList, packageName);
            System.out.println(file.getAbsolutePath() + "分析结束！");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void parseClassOrInterface(String prefix, BodyDeclaration classOrInterface,
                                       List<String> infoList) {
        ClassOrInterfaceDeclaration classInterface = (ClassOrInterfaceDeclaration)classOrInterface.toClassOrInterfaceDeclaration().get();
        // 过滤私有类或接口
        List<Modifier> classModifiers = classInterface.getModifiers();
        boolean classPrivateFlag = false;
        for (Modifier modifier : classModifiers) {
            if (modifier.getKeyword() == Modifier.Keyword.PRIVATE) {
                classPrivateFlag = true;
                break;
            }
        }
        if (classPrivateFlag) return;

        NodeList<BodyDeclaration<?>> list = classInterface.getMembers();
        for (BodyDeclaration body : list) {
            if (body.isClassOrInterfaceDeclaration())
                parseClassOrInterface(
                        prefix + "." + ((ClassOrInterfaceDeclaration)body.toClassOrInterfaceDeclaration().get()).getNameAsString(),
                        body, infoList);
            else if (body.isMethodDeclaration()) {
                CallableDeclaration method = (CallableDeclaration)body.toCallableDeclaration().get();
                // 过滤私有方法
                List<Modifier> modifiers = method.getModifiers();
                boolean privateFlag = false;
                for (Modifier modifier : modifiers) {
                    if (modifier.getKeyword() == Modifier.Keyword.PRIVATE) {
                        privateFlag = true;
                        break;
                    }
                }
                if (privateFlag) continue;

                // 处理方法文档注释，提取对应描述并预处理
                Javadoc javaDoc = (Javadoc)method.getJavadoc().orElse(new Javadoc(new JavadocDescription()));
                String description = ParserUtil.generateBriefDescription(javaDoc.getDescription().toText());
                description = ParserUtil.removeTag(description);
                description = ParserUtil.removeAtTag(description);
                description = ParserUtil.replaceHtmlChar(description);
                // 过滤没有注释的方法
                if (description == null || description.trim().equals(""))
                    continue;
                // 添加方法签名字符串
                StringBuilder strBuilder = new StringBuilder(prefix + "#");
                strBuilder.append(ParserUtil.generateMethodSignature(
                        method.getDeclarationAsString(false, false, false).trim(),
                        method.getNameAsString()));
                // 添加方法文档注释
                strBuilder.append(":::" + description);
                infoList.add(strBuilder.toString());
            }
        }
    }

    public void saveToFile(List<String> infoList, String packageName) {
        try {
            String path = resDataPath + File.separator + packageName + ".dat";
            BufferedWriter buffWriter = new BufferedWriter(new FileWriter(path, true));
            for (String info : infoList) {
                buffWriter.write(info);
                buffWriter.newLine();
            }
            buffWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
